import os
import json
import asyncio
import argparse
import pickle
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dataclasses import dataclass
import math

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from autogen_core.models import UserMessage, SystemMessage
from autogen_core import Image

import torch
from colpali_engine.models import ColPali, ColPaliProcessor

from prompts import PLANNER_SYSTEM_PROMPT, NAVIGATOR_VLM_PROMPT_TEMPLATE, REASONER_SYSTEM_PROMPT

import base64
import mimetypes
from openai import AsyncOpenAI

# ================= 配置管理 =================

@dataclass
class ModelConfig:
    api_key: str
    base_url: str
    model: str
    model_info: Dict[str, Any]

# 1. 核心推理模型配置 (Planner & Navigator)
# 如果是本地 vLLM，API Key 随便填，base_url 填本地地址
API_PLANNER_CONFIG = ModelConfig(
    api_key="sk-35093241c8aa4111862d53eeb486fdc0", # [填入你的 Key]
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # [填入对应 URL]
    model="qwen3-max", # [填入模型名称]
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown"
    }
)

# 2. 视觉模型配置 (Summarizer Tool)
# 专门用于看图的本地模型
API_REASONER_CONFIG = ModelConfig(
    api_key="sk-35093241c8aa4111862d53eeb486fdc0", # [填入你的 Key]
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # [填入对应 URL]
    model="qwen3-vl-plus-2025-12-19", # [填入模型名称]
    model_info={
        "vision": True,
        "function_calling": False,
        "json_output": False,
        "family": "unknown"
    }
)

LOCAL_NAVIGATOR_CONFIG = ModelConfig(
    api_key="sk-AL1JxJWykWTRZAIV4XGcmkhxGXXXa6WUt0S3uvs03AyLeWMj",
    base_url="https://api.kwwai.top/v1",
    model="gpt-5-mini", # 保持你本地的路径
    model_info={
        "vision": True,
        "function_calling": False,
        "json_output": False,
        "family": "unknown"
    }
)
# 3. ColPali 模型路径 (Retrieval)
COLPALI_MODEL_PATH = "/data/helq/doc_re/model/colpali-v1.2"

# ================= 系统环境类 =================

class BookEnvironment:
    """
    维护书籍状态、图谱和检索模型的单例环境。
    Tools 将会直接访问这个环境。
    """
    def __init__(self, asset_dir, image_root, device="cuda:1"):
        self.asset_dir = asset_dir
        self.image_root = image_root
        self.device = device
        
        print(f"Loading ColPali on {device}...")
        self.colpali = ColPali.from_pretrained(
            COLPALI_MODEL_PATH, dtype=torch.bfloat16, device_map=device, local_files_only=True
        ).eval()
        self.colpali_processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL_PATH, local_files_only=True)
        
        # 视觉客户端 (用于 Tool 内部)
        # self.vision_client = OpenAIChatCompletionClient(**LOCAL_NAVIGATOR_CONFIG.__dict__)
        self.vision_client = AsyncOpenAI(
            api_key=LOCAL_NAVIGATOR_CONFIG.api_key,
            base_url=LOCAL_NAVIGATOR_CONFIG.base_url
        )

        self.chapter_index = self._build_chapter_index()
        self.current_doc_id = None
        self.current_doc_embedding = None
        self.current_graph = None
        self._dir_cache = {}

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def _build_chapter_index(self):
        chapter_to_path = {}
        for book_name in os.listdir(self.image_root):
            book_path = os.path.join(self.image_root, book_name)
            if os.path.isdir(book_path):
                for chapter_name in os.listdir(book_path):
                    chapter_path = os.path.join(book_path, chapter_name)
                    if os.path.isdir(chapter_path):
                        chapter_to_path[chapter_name] = chapter_path
        return chapter_to_path

    def load_document(self, doc_id):
        if self.current_doc_id == doc_id: return
        print(f"Switching context to: {doc_id}")
        self.current_doc_id = doc_id
        emb_path = os.path.join(self.asset_dir, "embeddings", f"{doc_id}.pt")
        graph_path = os.path.join(self.asset_dir, "graphs", f"{doc_id}.pkl")
        
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embeddings not found for {doc_id}")
            
        self.current_doc_embedding = torch.load(emb_path, map_location="cpu").to(torch.bfloat16)
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                self.current_graph = pickle.load(f)
        else:
            self.current_graph = {}

    def get_image_path(self, page_idx):
        dir_path = self.chapter_index.get(self.current_doc_id)
        if not dir_path: return None
        if self.current_doc_id not in self._dir_cache:
            files = sorted([f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg'))], 
                           key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else 0)
            self._dir_cache[self.current_doc_id] = [os.path.join(dir_path, f) for f in files]
        files = self._dir_cache[self.current_doc_id]
        if 0 <= page_idx < len(files):
            return files[page_idx]
        return None

    def retrieve_initial_pages(self, query: str) -> List[int]:
        """
        Calculates similarity and returns the dynamic Top-K page indices (0-based).
        Logic: Top N = ceil(Total_Pages / 5).
        Returns indices sorted by similarity ASCENDING (low score -> high score),
        so that when pushed to stack, high score is at Top.
        """
        if self.current_doc_embedding is None:
            return []
        
        with torch.no_grad():
            batch_query = self.colpali_processor.process_queries([query]).to(self.device)
            query_emb = self.colpali(**batch_query)
            scores = self.colpali_processor.score_multi_vector(
                query_emb, self.current_doc_embedding.to(self.device)
            )
        
        # [修改] 返回所有页面的索引，按分数升序排列
        sorted_indices = scores[0].argsort(descending=False) # 升序
        return sorted_indices.tolist()

    def retrieve_page_scores(self, query: str) -> torch.Tensor:
        """
        Calculates similarity scores for all pages.
        Returns a tensor of scores [num_pages].
        """
        if self.current_doc_embedding is None:
            return torch.tensor([])
        
        with torch.no_grad():
            batch_query = self.colpali_processor.process_queries([query]).to(self.device)
            query_emb = self.colpali(**batch_query)
            scores = self.colpali_processor.score_multi_vector(
                query_emb, self.current_doc_embedding.to(self.device)
            )
        # scores shape: [1, num_pages] -> flatten to [num_pages]
        return scores[0].cpu()
    
    def get_semantic_neighbors(self, page_idx: int, k=3) -> List[int]:
        """获取 MoLoRAG 图中权重最高的 k 个邻居 (0-based)"""
        if not self.current_graph or page_idx not in self.current_graph:
            return []
        
        # current_graph 结构假设是 {page_idx: [neighbor_idx, ...]} 无权重，或者有权重
        # 如果你的 graph 只是邻接表 List，无法排序，直接返回
        neighbors = list(self.current_graph[page_idx])
        
        # 如果需要根据相似度排序，需要重新计算或读取边的权重。
        # 假设这里简单返回，或者如果 graph 存储了权重则按权重排序。
        # 暂时按默认顺序返回前 K 个
        return neighbors[:k]

# 全局实例，供 Tools 使用
book_env: Optional[BookEnvironment] = None

# ================= 工具定义 (Tools) =================
async def search_pages_tool(query: str) -> str:
    """Search for relevant pages in the current book chapter using keyword/semantic search."""
    global book_env
    if not book_env or book_env.current_doc_embedding is None:
        return "Error: No document loaded."
    
    print(f"  [Tool] Searching: {query}")
    with torch.no_grad():
        # 处理 Query
        batch_query = book_env.colpali_processor.process_queries([query]).to(book_env.device)
        query_emb = book_env.colpali(**batch_query)
        # 计算相似度
        scores = book_env.colpali_processor.score_multi_vector(
            query_emb, book_env.current_doc_embedding.to(book_env.device)
        )
    
    top_k_indices = scores[0].argsort(descending=True)[:5].tolist()
    results = [f"Page {idx + 1} (Score: {scores[0][idx]:.2f})" for idx in top_k_indices]
    return "\n".join(results)

async def get_neighbors_tool(page_num: int) -> str:
    """
    Get logical neighbors for a specific page number (1-based).
    Returns a list of connected page numbers (1-based).
    """
    global book_env
    if not book_env or book_env.current_doc_embedding is None: 
        return "Error: No doc loaded."

    current_idx = page_num - 1
    max_idx = len(book_env.current_doc_embedding) - 1
    
    if not (0 <= current_idx <= max_idx):
        return f"Error: Page {page_num} is out of range (1-{max_idx+1})."

    neighbors_idx = {current_idx - 1, current_idx + 1}
    
    # 从图谱中获取邻居 (图谱中存储的是 0-based 索引)
    if book_env.current_graph and current_idx in book_env.current_graph:
        neighbors_idx.update(book_env.current_graph[current_idx])

    valid_pages = [n + 1 for n in neighbors_idx if 0 <= n <= max_idx]
    
    return f"Page {page_num} connects to: {sorted(valid_pages)}"

# ================= 工具定义 (Tools: Modified to 1-based Indexing) =================

async def search_pages_tool(query: str) -> str:
    """
    Search for relevant pages. Returns page numbers (1-based) and relevance scores.
    """
    global book_env
    if not book_env or book_env.current_doc_embedding is None:
        return "Error: No document loaded."
    
    # print(f"  [Tool] Searching: {query}")
    with torch.no_grad():
        batch_query = book_env.colpali_processor.process_queries([query]).to(book_env.device)
        query_emb = book_env.colpali(**batch_query)
        scores = book_env.colpali_processor.score_multi_vector(
            query_emb, book_env.current_doc_embedding.to(book_env.device)
        )
    
    top_k_indices = scores[0].argsort(descending=True)[:5].tolist()
    
    # [修改点] idx + 1: 将 0-based 索引转换为 1-based 页码展示给 Agent
    results = [f"Page {idx + 1} (Score: {scores[0][idx]:.2f})" for idx in top_k_indices]
    return "\n".join(results)

async def get_neighbors_tool(page_num: int) -> str:
    """
    Get logical neighbors for a specific page number (1-based).
    Returns a list of connected page numbers (1-based).
    """
    global book_env
    if not book_env or book_env.current_doc_embedding is None: 
        return "Error: No doc loaded."
    
    # [修改点] 输入是 1-based 页码，转为 0-based 索引进行计算
    current_idx = page_num - 1
    max_idx = len(book_env.current_doc_embedding) - 1
    
    if not (0 <= current_idx <= max_idx):
        return f"Error: Page {page_num} is out of range (1-{max_idx+1})."

    neighbors_idx = {current_idx - 1, current_idx + 1}
    
    # 从图谱中获取邻居 (图谱中存储的是 0-based 索引)
    if book_env.current_graph and current_idx in book_env.current_graph:
        neighbors_idx.update(book_env.current_graph[current_idx])
    
    # [修改点] n + 1: 将合法的 0-based 邻居索引转回 1-based 页码
    valid_pages = [n + 1 for n in neighbors_idx if 0 <= n <= max_idx]
    
    return f"Page {page_num} connects to: {sorted(valid_pages)}"

async def read_page_tool(page_num: int, focus_query: str) -> str:
    """
    Read the visual content of a specific page number (1-based).
    """
    global book_env
    if not book_env: return "Error: Env not ready."
    img_path = book_env.get_image_path(page_num - 1)
    
    if not img_path:
        return f"Error: Image for Page {page_num} not found."
    
    print(f"  [Tool] VLM Reading Page {page_num} for: {focus_query}")
    user_prompt = f"Examine this image specifically looking for information about: '{focus_query}'. Summarize relevant details. If not present, say so."

    try:
        # [修改] 改为原生调用
        base64_image = book_env.encode_image(img_path)
        response = await book_env.vision_client.chat.completions.create(
            model=LOCAL_NAVIGATOR_CONFIG.model,
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }]
        )
        content = response.choices[0].message.content
        return f"Page {page_num} Analysis:\n{content}"
    except Exception as e:
        return f"Error calling vision model: {e}"

# ================= 智能体系统 (Agent Workflow) =================

class AgenticSystem:
    def __init__(self):
        # 初始化配置 (保持不变)
        self.planner_client = OpenAIChatCompletionClient(**API_PLANNER_CONFIG.__dict__)
        self.reasoner_client = OpenAIChatCompletionClient(**API_REASONER_CONFIG.__dict__)
        
    async def solve(self, question: str, options: List[str]) -> Dict:
        doc_logs = [] # 用于记录当前样本的详细日志
        
        # --- 1. Planner Phase ---
        planner = AssistantAgent(
            name="planner",
            model_client=self.planner_client, 
            system_message=PLANNER_SYSTEM_PROMPT
        )
        
        plan_res = await planner.on_messages(
            [TextMessage(content=f"Question: {question}", source="user")],
            cancellation_token=CancellationToken()
        )
        
        # 解析 Plan JSON
        raw_plan = plan_res.chat_message.content
        doc_logs.append(f"[Planner] Raw Output: {raw_plan}")
        try:
            # 尝试提取 JSON list
            match = re.search(r"\[.*\]", raw_plan, re.DOTALL)
            if match:
                search_steps = json.loads(match.group(0))
            else:
                # Fallback: 按行分割
                search_steps = [line.strip() for line in raw_plan.split('\n') if '-' in line or line[0].isdigit()]
        except:
            search_steps = [question] # Fallback

        # search_steps = [question]
            
        doc_logs.append(f"[Planner] Parsed Steps: {search_steps}")

        # --- 2. Navigator Phase (Algorithmic Loop) ---
        global_evidence_pool = [] # 存储 (page_num, text_summary)
        all_retrieved_pages = set() # 存储所有相关页面的 ID (1-based) for sampling
        global_score_accumulator = None
        
        # 跨 Step 的黑名单 (如果在 Step1 被认定无关，Step2 也不用看了吗？
        # 通常建议每个 Step 独立黑名单，因为 Query 不同。
        # 但你提到“被排除过的页面不再考虑”，这里我们暂且维护一个 step-specific blacklist)
        
        for step_idx, step_query in enumerate(search_steps):
            doc_logs.append(f"\n--- Processing Step {step_idx+1}: {step_query} ---")
            
            current_scores = book_env.retrieve_page_scores(step_query)
            if len(current_scores) > 0:
                if global_score_accumulator is None:
                    global_score_accumulator = torch.zeros_like(current_scores)
                # 简单相加 (Sum Fusion)
                global_score_accumulator += current_scores

            # 2.1 初始化状态
            step_accepted_pages = [] # 存 0-based index
            step_blacklist = set()   # 存 0-based index
            search_stack = []        # 存 0-based index
            
            # 2.2 初始入栈 (ColPali)
            # 返回的是 [Low Score, ..., High Score]
            initial_pages = book_env.retrieve_initial_pages(step_query)
            search_stack.extend(initial_pages)
            
            stack_preview = [p+1 for p in search_stack[-5:]] # 只看栈顶5个
            doc_logs.append(f"Initial Stack Size: {len(search_stack)}, Top: {stack_preview}")

            # 2.3 搜索循环
            while len(step_accepted_pages) < 3 and search_stack:
                # Pop 栈顶 (ColPali 这里的栈顶是相似度最高的)
                current_idx = search_stack.pop()
                current_page_num = current_idx + 1 # 1-based for logs
                
                # Check Filters
                if current_idx in step_blacklist:
                    continue
                if current_idx in step_accepted_pages:
                    continue
                # 也要检查是否在之前 Step 已经判定为相关（避免重复阅读？）
                # 这里策略是：即使之前相关，针对新 Query 也要重新看一遍提取新信息
                
                # VLM Analysis
                img_path = book_env.get_image_path(current_idx)
                if not img_path: continue

                # 构造 VLM Prompt
                vlm_prompt = NAVIGATOR_VLM_PROMPT_TEMPLATE.format(query=step_query)
                # vlm_msg = UserMessage(content=[vlm_prompt, Image.from_file(img_path)], source="user")
                
                # try:
                #     # 直接调用 Vision Client
                #     response = await book_env.vision_client.create([vlm_msg])
                #     vlm_content = response.content
                #     doc_logs.append(f"[VLM Analysis Page {current_page_num}]: {vlm_content}")
                # except Exception as e:
                #     doc_logs.append(f"Error reading page {current_page_num}: {e}")
                #     continue
                try:
                    base64_image = book_env.encode_image(img_path)
                    mime_type, _ = mimetypes.guess_type(img_path)
                    if not mime_type: mime_type = "image/jpeg"

                    response = await book_env.vision_client.chat.completions.create(
                        model=LOCAL_NAVIGATOR_CONFIG.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": vlm_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=1024
                    )
                    
                    # 3. [核心修复] 防御性解析 (Polymorphic Parsing)
                    vlm_content = ""
                    
                    # 情况 A: response 是个字符串 (目前的报错原因)
                    if isinstance(response, str):
                        # 尝试把它当做 JSON 字符串解析
                        try:
                            data = json.loads(response)
                            if isinstance(data, dict) and "choices" in data:
                                # 成功解析出 JSON 结构
                                vlm_content = data["choices"][0]["message"]["content"]
                            else:
                                # 解析了但没有 choices，或者它就是纯文本回复
                                vlm_content = response 
                        except json.JSONDecodeError:
                            # 根本不是 JSON，那它可能就是纯文本回复（或者报错信息）
                            vlm_content = response

                    # 情况 B: response 是标准的 OpenAI 对象 (含有 choices 属性)
                    elif hasattr(response, "choices"):
                        vlm_content = response.choices[0].message.content
                        
                    # 情况 C: response 是个字典 (旧版 SDK 或 requests 返回)
                    elif isinstance(response, dict) and "choices" in response:
                        vlm_content = response["choices"][0]["message"]["content"]
                        
                    else:
                        # 兜底
                        vlm_content = str(response)

                    # 记录日志
                    doc_logs.append(f"[VLM Analysis Page {current_page_num}]: {vlm_content}")

                except Exception as e:
                    # 打印详细报错堆栈，方便看清到底是网络问题还是解析问题
                    import traceback
                    print(f"!!! Error reading page {current_page_num}: {e}")
                    traceback.print_exc()
                    doc_logs.append(f"Error reading page {current_page_num}: {e}")
                    continue

                # 判断相关性
                if "[IRRELEVANT]" in vlm_content:
                    step_blacklist.add(current_idx)
                    doc_logs.append(f"[Navigator] Page {current_page_num} -> IRRELEVANT. Added to Blacklist.")
                else:
                    # 判定为相关
                    doc_logs.append(f"[Navigator] Page {current_page_num} -> RELEVANT. Accepted.")
                    step_accepted_pages.append(current_idx)
                    
                    # 存入全局证据
                    evidence_entry = f"Evidence from Page {current_page_num} regarding '{step_query}':\n{vlm_content}"
                    global_evidence_pool.append(evidence_entry)
                    all_retrieved_pages.add(current_idx) # 0-based set
                    
                    # --- 邻居扩展 (Expansion) ---
                    # 目标出栈顺序：上一页 -> 下一页 -> 图谱邻居
                    # 意味着入栈顺序：图谱 -> 下一页 -> 上一页
                    
                    neighbors_candidate = []
                    
                    # 1. Graph Neighbors (最底层)
                    graph_neighbors = book_env.get_semantic_neighbors(current_idx, k=3)
                    neighbors_candidate.extend(graph_neighbors)
                    
                    # 2. Prev Page
                    if current_idx - 1 >= 0:
                        neighbors_candidate.append(current_idx - 1)
                        
                    # 3. Next Page (最后压入，最先 Pop)
                    if current_idx + 1 < len(book_env.current_doc_embedding):
                        neighbors_candidate.append(current_idx + 1)
                    
                        
                    # 执行压栈 (需检查去重)
                    for n_idx in neighbors_candidate:
                        # 1. 过滤掉已经处理过的（黑名单或已接受）
                        if n_idx in step_blacklist or n_idx in step_accepted_pages:
                            continue
                        
                        # 2. [关键修复] 如果已经在栈中（比如在栈底），先删除它
                        # 这样我们稍后 append 时，它就会跑到栈顶，被优先考察
                        if n_idx in search_stack:
                            search_stack.remove(n_idx)
                        
                        # 3. 压入栈顶
                        search_stack.append(n_idx)
                             
                    stack_top_preview = [p+1 for p in search_stack[-5:]] # 1-based
                    doc_logs.append(f"  > Expanded neighbors for Page {current_page_num}. New Stack Top (Right is Top): {stack_top_preview}")
                doc_logs.append(f"  [State] Accepted: {[p+1 for p in step_accepted_pages]}, Blacklist: {len(step_blacklist)}")
            doc_logs.append(f"\n[Step End] Global Evidence Pool: {len(global_evidence_pool)} items")
            doc_logs.append(f"[Step End] Current All Retrieved Pages: {[p+1 for p in all_retrieved_pages]}")

        # --- 3. Reasoner Phase ---
        
        # 3.1 公平采样 (Round-Robin Sampling for Images)
        # 我们需要把 all_retrieved_pages 对应的图片喂给 Reasoner
        # 但 VLM 有输入限制 (如 10 张)。
        # 策略：按 Step 收集的页面进行轮询
        
        agent_selected_indices = list(all_retrieved_pages)
        filled_indices = []
        target_count = 10

        # 情况 A: Agent 找太多了 -> 截断 (优先保留 Agent 的结果)
        if len(agent_selected_indices) > target_count:
             # 简单按页码排序截断，或者你可以保留先发现的。这里按页码排序稳健。
             agent_selected_indices = sorted(agent_selected_indices)[:target_count]
             doc_logs.append(f"[Sampling] Truncated Agent-selected pages to {target_count}.")
        
        # 情况 B: 不够 10 页 -> 补齐
        elif len(agent_selected_indices) < target_count and global_score_accumulator is not None:
            needed = target_count - len(agent_selected_indices)
            doc_logs.append(f"[Sampling] Need to fill {needed} pages. Using Global Fusion Scores.")
            
            # 按分数降序排列所有页面
            sorted_indices_by_score = global_score_accumulator.argsort(descending=True).tolist()
            
            filled_count = 0
            for idx in sorted_indices_by_score:
                if filled_count >= needed:
                    break
                # 如果这个高分页面既没被 Agent 选中，也没被加入补齐列表
                if idx not in all_retrieved_pages and idx not in filled_indices:
                    filled_indices.append(idx)
                    filled_count += 1
            
            doc_logs.append(f"[Sampling] Filled {filled_count} pages based on global scores.")

        # [关键修改] 构造最终列表：Agent 确认页在前 + 补齐页在后
        # 组内按页码排序，符合人类阅读习惯
        final_image_indices = sorted(agent_selected_indices) + sorted(filled_indices)
        
        # 为了防止意外（比如 all_retrieved_pages 没去重），再做一次去重检查（理论上上面逻辑已保证）
        # final_image_indices = list(dict.fromkeys(final_image_indices)) 
        
        doc_logs.append(f"[Sampling] Final Context Pages: {[i+1 for i in final_image_indices]}")
        
        # 3.2 构造 Reasoner Input
        reasoner_content = [
            REASONER_SYSTEM_PROMPT,
            f"Question: {question}\nOptions: {options}",
            "--- Collected Text Evidence ---",
            "\n\n".join(global_evidence_pool),
            "--- Visual Evidence (Reference Pages) ---"
        ]
        
        # 添加图片
        for idx in final_image_indices:
            path = book_env.get_image_path(idx)
            if path:
                reasoner_content.append(f"Page {idx+1}")
                reasoner_content.append(Image.from_file(path))
        
        reasoner_msg = UserMessage(content=reasoner_content, source="user")
        
        # 调用 Reasoner
        try:
            # 这里直接用 client 调用，不走 Agent 也可以，或者走 Agent
            # 为了保持一致性，用 client 直接生成
            response = await self.reasoner_client.create([reasoner_msg])
            pred_text = response.content
        except Exception as e:
            pred_text = "Error"
            doc_logs.append(f"[Reasoner Error] {e}")

        # 提取答案
        pred_answer = "N/A"
        explicit_match = re.search(r"Final Answer:\s*([A-F])", pred_text, re.IGNORECASE)

        if explicit_match:
            pred_answer = explicit_match.group(1).upper()
        else:
            # 策略 2: 如果没有标准格式，查找文中出现的 *最后一个* 选项字母
            # re.findall 会返回列表，[-1] 取最后一个
            candidates = re.findall(r"\b([A-F])\b", pred_text.upper())
            if candidates:
                pred_answer = candidates[-1]
                doc_logs.append("[Reasoner] Warning: Specific format not found, picked the LAST option letter.")
        
        doc_logs.append(f"[Reasoner] Output: {pred_text} -> Prediction: {pred_answer}")
        
        # 返回结果 (包含日志以便写入文件)
        return {
            "pred_answer": pred_answer,
            "retrieved_indices": final_image_indices,
            "logs": "\n".join(doc_logs)
        }

# ================= 主流程 (Main) =================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/helq/doc_re/m3bookvqa/m3bookvqa/data.jsonl")
    parser.add_argument("--image_root", type=str, default="/data/helq/doc_re/m3bookvqa/imgs")
    parser.add_argument("--molorag_dir", type=str, default="/data/helq/doc_re/m3bookvqa/molorag")
    parser.add_argument("--output_file", type=str, default="/data/helq/doc_re/m3bookvqa/agenticRAG/results_ablation_tracker/results1400_2054.json")
    args = parser.parse_args()

    raw_log_dir = os.path.join(os.path.dirname(args.output_file), "raw")
    os.makedirs(raw_log_dir, exist_ok=True)

    # 初始化全局环境
    global book_env
    book_env = BookEnvironment(args.molorag_dir, args.image_root)
    
    # 初始化 Agent 系统
    system = AgenticSystem()

    # 读取数据
    data = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))

    results = []
    
    for sample in tqdm(data[1400:]):
        doc_id = sample['id'].rsplit('-', 1)[0]
        # 加载书本资源
        book_env.load_document(doc_id)
        
        try:
            # solve 现在返回一个 dict
            result_pkg = await system.solve(sample['question'], sample['options'])
            pred = result_pkg["pred_answer"]
            ret_indices = result_pkg["retrieved_indices"] # 0-based list
            logs = result_pkg["logs"]
            
            # 5. 保存详细日志 txt
            log_path = os.path.join(raw_log_dir, f"{sample['id']}.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(logs)
                
        except Exception as e:
            print(f"Error: {e}")
            pred = "Error"
            ret_indices = []

        retrieved_uids = [f"{doc_id}_page{idx+1}" for idx in ret_indices]
            
        print(f"GT: {sample['answer']} | Pred: {pred}")
        
        gt_uids = [f"{doc_id}_page{p_num}" for p_num in sample['page_numbers']]

        res_entry = {
            "id": sample['id'],
            "question": sample['question'],
            "ground_truth_answer": sample['answer'],
            "predicted_answer": pred,
            "ground_truth_pages": gt_uids,
            "raw_ground_truth_pages": sample['page_numbers'],
            "retrieved_pages": retrieved_uids,
            "options": sample['options']
        }
        
        results.append(res_entry)
        
        # 实时保存
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())

