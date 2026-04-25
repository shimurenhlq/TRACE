import os
import json
import torch
import pickle
import math
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from bisect import bisect_right

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from autogen_core.models import UserMessage
from autogen_core import Image

from colpali_engine.models import ColPali, ColPaliProcessor
from prompts import PLANNER_SYSTEM_PROMPT, NAVIGATOR_VLM_PROMPT_TEMPLATE, REASONER_SYSTEM_PROMPT

# ================= 配置管理 =================

@dataclass
class ModelConfig:
    api_key: str
    base_url: str
    model: str
    model_info: Dict[str, Any]

# 1. 规划师 (Planner) - 外部 API
API_PLANNER_CONFIG = ModelConfig(
    api_key="sk-3ea10b3683a749c49499c615e5616cea",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-max",
    model_info={"vision": False, "function_calling": True, "json_output": True, "family": "unknown"}
)

# 2. 推理师 (Reasoner) - 外部 API (长窗口 VLM)
API_REASONER_CONFIG = ModelConfig(
    api_key="sk-3ea10b3683a749c49499c615e5616cea",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-vl-plus-2025-12-19",
    model_info={"vision": True, "function_calling": False, "json_output": False, "family": "unknown"}
)

# 3. [新增] 导航员 (Navigator) - 外部 API (快速 VLM)
API_NAVIGATOR_CONFIG = ModelConfig(
    api_key="sk-3ea10b3683a749c49499c615e5616cea",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-vl-8b-instruct", # 阿里云支持此模型名
    model_info={"vision": True, "function_calling": False, "json_output": False, "family": "unknown"}
)

# ColPali 配置
COLPALI_MODEL_PATH = "/data/helq/doc_re/model/colpali-v1.2"
COLPALI_DEVICE = "cpu" # [指定] 运行在 CUDA:1

# ================= 环境类 (核心重构) =================

class BookEnvironment:
    def __init__(self, asset_dir, image_root):
        self.asset_dir = asset_dir
        self.image_root = image_root
        self.device = COLPALI_DEVICE
        
        print(f"Loading ColPali on {self.device}...")
        self.colpali = ColPali.from_pretrained(
            COLPALI_MODEL_PATH, dtype=torch.bfloat16, device_map=self.device, local_files_only=True
        ).eval()
        self.colpali_processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL_PATH, local_files_only=True)
        
        # [修改] 使用 API 配置初始化 Navigator 的视觉客户端
        self.vision_client = OpenAIChatCompletionClient(**API_NAVIGATOR_CONFIG.__dict__)

        # 构建 doc_id -> image_dir 的映射
        self.chapter_index = self._build_chapter_index()
        
        # 资源缓存
        self.current_group_id = None
        self.current_embeddings = None # 大矩阵
        self.current_graph = None
        self.meta_segments = []       # 索引映射表
        self.meta_starts = []         # 用于二分查找

    def _build_chapter_index(self):
        """遍历 imgs 目录，构建 章节名 -> 图片目录 的映射"""
        chapter_to_path = {}
        if not os.path.exists(self.image_root):
            return {}
        for book_name in os.listdir(self.image_root):
            book_path = os.path.join(self.image_root, book_name)
            if os.path.isdir(book_path):
                for chapter_name in os.listdir(book_path):
                    chapter_path = os.path.join(book_path, chapter_name)
                    if os.path.isdir(chapter_path):
                        chapter_to_path[chapter_name] = chapter_path
        return chapter_to_path

    def load_resources(self, group_id: str, mode: str = "book"):
        """
        加载指定范围的资源 (Book or Global)
        group_id: 书名 (mode='book') 或 'global' (mode='global')
        """
        if self.current_group_id == group_id:
            return

        print(f"Loading resources for group: {group_id} ({mode})...")
        self.current_group_id = group_id
        
        # 确定路径前缀
        prefix = "book" if mode == "book" else "global"
        
        # 1. 加载 Embeddings
        emb_path = os.path.join(self.asset_dir, f"{prefix}_embeddings", f"{group_id}.pt")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embeddings not found: {emb_path}")
        # 加载到 CPU，计算时再挪到 GPU，节省显存
        self.current_embeddings = torch.load(emb_path, map_location="cpu").to(torch.bfloat16)

        # 2. 加载 Graph
        graph_path = os.path.join(self.asset_dir, f"{prefix}_graphs", f"{group_id}.pkl")
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                self.current_graph = pickle.load(f)
        else:
            self.current_graph = {}

        # 3. 加载 Meta 映射文件
        # 假设 prep_m3book_data 生成了对应名字的 _meta.json
        # 如果是 book 模式: book_embeddings/{book_name}_meta.json (根据你的 prep 代码逻辑调整)
        # 如果是 global 模式: global_embeddings/global_meta.json
        meta_filename = f"{group_id}_meta.json"
        meta_path = os.path.join(self.asset_dir, f"{prefix}_embeddings", meta_filename)
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
            
        with open(meta_path, 'r') as f:
            self.meta_segments = json.load(f)
            
        # 预计算 start 列表用于二分查找
        self.meta_starts = [seg['start'] for seg in self.meta_segments]

    def _resolve_global_idx(self, global_idx: int) -> Tuple[str, int]:
        """将全局索引映射回 (doc_id, local_page_idx)"""
        # 使用 bisect_right 找到插入点
        # 例如 starts=[0, 10, 30], idx=15. bisect_right 返回 2 (对应 30). 
        # segment index = 2-1 = 1 (对应 10).
        seg_idx = bisect_right(self.meta_starts, global_idx) - 1
        if seg_idx < 0 or seg_idx >= len(self.meta_segments):
            return None, -1
            
        segment = self.meta_segments[seg_idx]
        # 校验范围
        if not (segment['start'] <= global_idx < segment['start'] + segment['length']):
            return None, -1
            
        local_page_idx = global_idx - segment['start'] # 0-based
        return segment['doc_id'], local_page_idx

    def get_image_path(self, global_idx: int) -> Optional[str]:
        """根据全局索引获取图片路径"""
        doc_id, local_page_idx = self._resolve_global_idx(global_idx)
        if not doc_id:
            return None
            
        chapter_dir = self.chapter_index.get(doc_id)
        if not chapter_dir:
            return None
            
        # 构造文件名: {doc_id}_{page_num}.png (page_num 是 1-based)
        img_name = f"{doc_id}_{local_page_idx + 1}.png"
        full_path = os.path.join(chapter_dir, img_name)
        
        if os.path.exists(full_path):
            return full_path
        # 尝试 .jpg
        full_path = full_path.replace(".png", ".jpg")
        if os.path.exists(full_path):
            return full_path
            
        return None

    def index_to_uid(self, global_idx: int) -> str:
        """将索引转换为结果记录用的 UID 格式"""
        doc_id, local_page_idx = self._resolve_global_idx(global_idx)
        if doc_id:
            return f"{doc_id}_page{local_page_idx + 1}"
        return f"unknown_{global_idx}"

    def retrieve_page_scores(self, query: str) -> torch.Tensor:
        """计算全量页面的相似度分数"""
        if self.current_embeddings is None:
            return torch.tensor([])
        
        with torch.no_grad():
            batch_query = self.colpali_processor.process_queries([query]).to(self.device)
            query_emb = self.colpali(**batch_query)
            
            # 将大矩阵移动到 GPU 进行计算 (如果显存不够，可以分块计算)
            # 假设 embeddings 放在 cpu，这里临时 to(device)
            doc_emb_device = self.current_embeddings.to(self.device)
            
            scores = self.colpali_processor.score_multi_vector(
                query_emb, doc_emb_device
            )
            
        return scores[0].cpu() # 返回 CPU tensor

    def retrieve_initial_pages(self, query: str) -> List[int]:
        """返回按分数升序排列的所有索引"""
        scores = self.retrieve_page_scores(query)
        if len(scores) == 0: return []
        sorted_indices = scores.argsort(descending=False)
        return sorted_indices.tolist()

    def get_semantic_neighbors(self, global_idx: int, k=3) -> List[int]:
        """从图谱中获取邻居 (输入输出均为全局索引)"""
        if not self.current_graph or global_idx not in self.current_graph:
            return []
        # 假设图谱存的是邻接列表
        neighbors = list(self.current_graph[global_idx])
        return neighbors[:k]

# ================= 智能体系统 =================

class AgenticSystem:
    def __init__(self):
        self.planner_client = OpenAIChatCompletionClient(**API_PLANNER_CONFIG.__dict__)
        self.reasoner_client = OpenAIChatCompletionClient(**API_REASONER_CONFIG.__dict__)
        # Navigator client 已集成在 BookEnvironment 中
        
    async def solve(self, question: str, options: List[str], book_env: BookEnvironment) -> Dict:
        # 注意：这里需要传入 book_env 实例，因为它是 solve 的上下文
        doc_logs = []
        
        # --- 1. Planner ---
        planner = AssistantAgent(
            name="planner",
            model_client=self.planner_client, 
            system_message=PLANNER_SYSTEM_PROMPT
        )
        try:
            plan_res = await planner.on_messages(
                [TextMessage(content=f"Question: {question}", source="user")],
                cancellation_token=CancellationToken()
            )
            raw_plan = plan_res.chat_message.content
            doc_logs.append(f"[Planner] Raw: {raw_plan}")
            match = re.search(r"\[.*\]", raw_plan, re.DOTALL)
            search_steps = json.loads(match.group(0)) if match else [question]
        except:
            search_steps = [question]
            
        doc_logs.append(f"[Planner] Steps: {search_steps}")

        # --- 2. Navigator Loop ---
        global_evidence_pool = []
        all_retrieved_pages = set()
        global_score_accumulator = None
        
        for step_idx, step_query in enumerate(search_steps):
            doc_logs.append(f"\n--- Step {step_idx+1}: {step_query} ---")
            
            # 计算分数
            current_scores = book_env.retrieve_page_scores(step_query)
            if len(current_scores) > 0:
                if global_score_accumulator is None:
                    global_score_accumulator = torch.zeros_like(current_scores)
                global_score_accumulator += current_scores
            
            step_accepted = []
            step_blacklist = set()
            search_stack = book_env.retrieve_initial_pages(step_query) # 升序，高分在后
            
            # 简单展示栈顶
            doc_logs.append(f"Stack Top: {[book_env.index_to_uid(i) for i in search_stack[-3:]]}")

            while len(step_accepted) < 3 and search_stack:
                curr_idx = search_stack.pop()
                curr_uid = book_env.index_to_uid(curr_idx) # [新增] 获取可读 ID
                
                if curr_idx in step_blacklist or curr_idx in step_accepted: continue
                
                img_path = book_env.get_image_path(curr_idx)
                if not img_path: continue
                
                # Navigator 视觉分析
                vlm_msg = UserMessage(content=[
                    NAVIGATOR_VLM_PROMPT_TEMPLATE.format(query=step_query),
                    Image.from_file(img_path)
                ], source="user")
                
                try:
                    resp = await book_env.vision_client.create([vlm_msg])
                    content = resp.content
                    doc_logs.append(f"[VLM {curr_uid}]: {content}")
                except Exception as e:
                    doc_logs.append(f"VLM Error: {e}")
                    continue
                
                if "[IRRELEVANT]" in content:
                    step_blacklist.add(curr_idx)
                    # [新增] 记录无关判定
                    doc_logs.append(f"[Navigator] {curr_uid} -> IRRELEVANT. Added to Blacklist.")
                else:
                    # [新增] 记录相关判定
                    doc_logs.append(f"[Navigator] {curr_uid} -> RELEVANT. Accepted.")
                    step_accepted.append(curr_idx)
                    all_retrieved_pages.add(curr_idx)
                    global_evidence_pool.append(f"Page {curr_uid}: {content}")
                    
                    # 邻居扩展
                    neighbors = []
                    neighbors.extend(book_env.get_semantic_neighbors(curr_idx, k=3))
                    # 全局索引的下一页
                    if curr_idx + 1 < len(book_env.current_embeddings): neighbors.append(curr_idx + 1)
                    # 全局索引的上一页
                    if curr_idx - 1 >= 0: neighbors.append(curr_idx - 1)
                    
                    for n in neighbors:
                        if n in step_blacklist or n in step_accepted: continue
                        if n in search_stack: search_stack.remove(n)
                        search_stack.append(n) # 移至栈顶
                    
                    # [新增] 记录邻居扩展后的栈顶状态
                    stack_preview = [book_env.index_to_uid(i) for i in search_stack[-3:]]
                    doc_logs.append(f"  > Expanded neighbors. New Stack Top: {stack_preview}")

                # [新增] 记录当前 Step 的状态（接受列表和黑名单数量）
                accepted_uids = [book_env.index_to_uid(i) for i in step_accepted]
                doc_logs.append(f"  [State] Accepted: {accepted_uids}, Blacklist Count: {len(step_blacklist)}")

            # [新增] Step 结束时的汇总记录
            all_retrieved_uids = [book_env.index_to_uid(i) for i in all_retrieved_pages]
            doc_logs.append(f"\n[Step End] Global Evidence Pool: {len(global_evidence_pool)} items")
            doc_logs.append(f"[Step End] Current All Retrieved Pages: {all_retrieved_uids}")

        # --- 3. Reasoner & Sampling ---
        agent_indices = list(all_retrieved_pages)
        filled_indices = []
        target_count = 10
        
        if len(agent_indices) > target_count:
            agent_indices = sorted(agent_indices)[:target_count]
        elif len(agent_indices) < target_count and global_score_accumulator is not None:
            needed = target_count - len(agent_indices)
            # 降序
            sorted_score_idx = global_score_accumulator.argsort(descending=True).tolist()
            for idx in sorted_score_idx:
                if len(filled_indices) >= needed: break
                if idx not in all_retrieved_pages:
                    filled_indices.append(idx)
        
        final_indices = sorted(agent_indices) + sorted(filled_indices)
        
        # 构造 Reasoner 输入
        reasoner_content = [
            REASONER_SYSTEM_PROMPT,
            f"Question: {question}\nOptions: {options}",
            "--- Evidence ---", "\n".join(global_evidence_pool),
            "--- Reference Images ---"
        ]
        for idx in final_indices:
            p = book_env.get_image_path(idx)
            if p:
                reasoner_content.append(f"Page {book_env.index_to_uid(idx)}")
                reasoner_content.append(Image.from_file(p))
                
        try:
            resp = await self.reasoner_client.create([UserMessage(content=reasoner_content, source="user")])
            pred_text = resp.content
        except Exception as e:
            pred_text = "Error"
            doc_logs.append(f"Reasoner Error: {e}")
            
        match = re.search(r"Final Answer:\s*([A-F])", pred_text, re.IGNORECASE)
        pred_ans = match.group(1).upper() if match else "N/A"
        if pred_ans == "N/A":
             cands = re.findall(r"\b([A-F])\b", pred_text.upper())
             if cands: pred_ans = cands[-1]

        return {
            "pred_answer": pred_ans,
            "retrieved_indices": final_indices,
            "logs": "\n".join(doc_logs)
        }