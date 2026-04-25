import os
import json
import torch
import numpy as np
import pickle
from tqdm import tqdm
from utils.datautil import construct_page_graph
# 假设你已经安装了 colpali_engine 和相关依赖
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image
import argparse

def prepare_data(data_path, image_root, output_dir, device="cuda"):
    os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)
    
    # 1. 加载模型 (ColPali) - 强制离线模式
    print("Loading ColPali model...")
    model_name = "/data/helq/doc_re/model/colpali-v1.2" # 或你使用的版本
    model = ColPali.from_pretrained(
        model_name, 
        dtype=torch.bfloat16, 
        device_map=device, 
        local_files_only=True
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name, local_files_only=True)

    # 2. 读取原始数据
    data = []
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))

    # 3. 构建章节索引 (Chapter Name -> Full Path)
    print("Building Chapter Index from images directory...")
    chapter_to_path = {}
    
    # 遍历 imgs 目录下的所有书籍文件夹
    if not os.path.exists(image_root):
        raise FileNotFoundError(f"Image root directory not found: {image_root}")

    for book_name in os.listdir(image_root):
        book_path = os.path.join(image_root, book_name)
        if os.path.isdir(book_path):
            # 遍历书籍下的所有章节文件夹
            for chapter_name in os.listdir(book_path):
                chapter_path = os.path.join(book_path, chapter_name)
                if os.path.isdir(chapter_path):
                    # 如果有重名章节，这里会发生覆盖。请确保章节名唯一。
                    chapter_to_path[chapter_name] = chapter_path
    
    print(f"Index built. Found {len(chapter_to_path)} chapters.")
    
    processed_docs = set()
    
    print("Processing Documents (Embedding & Graph Construction)...")
    for sample in tqdm(data):
        # 解析 doc_id (章节名)
        # 根据你的说明：id由 "章节名-问题序号" 组成
        doc_id = sample['id'].rsplit('-', 1)[0] 
        
        if doc_id in processed_docs:
            continue
            
        # --- 严格校验：必须找到对应路径 ---
        doc_img_dir = chapter_to_path.get(doc_id)
        
        if not doc_img_dir:
            # 报错：数据集中有的章节，在图片目录索引中没找到
            raise ValueError(f"Dataset contains chapter '{doc_id}' (from sample id '{sample['id']}'), but it was NOT found in image_root '{image_root}'.")
        
        if not os.path.exists(doc_img_dir):
            # 报错：路径虽然在索引里（理论上不应该发生），但在磁盘上消失了
            raise FileNotFoundError(f"Directory for chapter '{doc_id}' does not exist at: {doc_img_dir}")
            
        # 读取图片
        image_files = sorted([f for f in os.listdir(doc_img_dir) if f.endswith(('.png', '.jpg'))], 
                             key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else 0)
        
        image_paths = [os.path.join(doc_img_dir, f) for f in image_files]
        
        if len(image_paths) == 0:
            # 报错：文件夹存在，但是里面没有图片
            raise ValueError(f"Chapter directory for '{doc_id}' is empty (no .png/.jpg found): {doc_img_dir}")

        # --- A. 计算 Embeddings ---
        emb_path = os.path.join(output_dir, "embeddings", f"{doc_id}.pt")
        
        # 只有当 embedding 不存在时才计算
        # 只有当 embedding 不存在时才计算
        if not os.path.exists(emb_path):
            # 将路径转为 PIL Image 对象
            loaded_images = [Image.open(p).convert("RGB") for p in image_paths]
            
            all_embs = []
            batch_size = 1  # 设定显存允许的 Batch Size
            
            with torch.no_grad():
                for i in range(0, len(loaded_images), batch_size):
                    batch_imgs = loaded_images[i : i + batch_size]
                    batch_inputs = processor.process_images(batch_imgs)
                    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                    embs = model(**batch_inputs)
                    all_embs.append(embs.cpu())
            
            if all_embs:
                doc_embedding = torch.cat(all_embs, dim=0)
                torch.save(doc_embedding, emb_path)
            else:
                raise RuntimeError(f"Failed to generate embeddings for '{doc_id}'.")
        else:
            # 如果已存在，加载以便后续构建图
            doc_embedding = torch.load(emb_path)
            
        # --- B. 构建 Page Graph (MoLoRAG 关键步骤) ---
        graph_path = os.path.join(output_dir, "graphs", f"{doc_id}.pkl")
        
        if not os.path.exists(graph_path):
            print(f"Constructing graph for {doc_id}...")
            # --- 修改点 2: 移除 mean(dim=1) 和转置 ---
            # 官方 construct_page_graph 接收 [N, 32, 128] 的 doc_embedding
            # 并内部调用 compute_embed_similarity 处理多向量相似度
            
            # 注意：需确保 doc_embedding 转为 numpy (如果 utils.datautil 需要 numpy)
            # 或者是 Tensor (如果 utils 内部用 torch 计算)。
            # 查看官方代码 utils/datautil.py: 
            # vec_i, vec_j = doc_emb[i], doc_emb[j]
            # compute_embed_similarity 通常处理 Tensor 或 Numpy。
            # 为了安全，官方 load_all_doc_embeddings 转成了 numpy，我们这里也转
            doc_emb_input = doc_embedding.float().numpy()
            
            # 官方默认 threshold=0.7 (你之前是0.5，建议对齐官方或根据书籍微调)
            graph = construct_page_graph(doc_emb_input, threshold=0.7, k_value=5)
            
            with open(graph_path, 'wb') as f:
                pickle.dump(graph, f)
        
        processed_docs.add(doc_id)

    print(f"Data preparation complete. Processed {len(processed_docs)} documents.")

def build_graph_gpu_fast(embeddings, threshold=0.7, k_value=5, device="cuda"):
    """
    使用 Mean Pooling + Cosine Similarity 快速构建大图
    与 utils.datautil.construct_page_graph 逻辑对齐，但使用 GPU 加速 O(N^2)
    """
    print(f"Building fast graph on GPU for {embeddings.shape[0]} pages...")
    n_pages = embeddings.shape[0]
    
    # 1. Mean Pooling: [N, 32, 128] -> [N, 128]
    # 使用 float32 保证精度
    embs_mean = embeddings.mean(dim=1).to(device, dtype=torch.float32)
    
    # 2. L2 Normalize (以便后续直接用 dot product 计算 Cosine)
    embs_mean = torch.nn.functional.normalize(embs_mean, p=2, dim=1)
    
    # 3. 计算 Cosine Matrix: [N, N]
    # matrix[i, j] = dot(e_i, e_j)
    sim_matrix = torch.matmul(embs_mean, embs_mean.T) # [N, N]
    
    # 4. 构建邻接表 (转回 CPU 处理索引，因为 graph dict 是 python 对象)
    # 对于每一行，找 top-k 且 > threshold
    # 可以在 GPU 上做 TopK
    
    # values: [N, K], indices: [N, K]
    topk_values, topk_indices = torch.topk(sim_matrix, k=k_value, dim=1)
    
    topk_values = topk_values.cpu().numpy()
    topk_indices = topk_indices.cpu().numpy()
    from collections import defaultdict
    graph = defaultdict(list)
    
    # 遍历填入 Graph
    for i in tqdm(range(n_pages)):
        for k in range(k_value):
            score = topk_values[i, k]
            neighbor = topk_indices[i, k]
            
            # 排除自环 (虽然 topk 肯定包含自己，通常是第一个)
            if neighbor == i:
                continue
                
            if score >= threshold:
                graph[int(i)].append(int(neighbor))
                # 无向图，对称添加? 官方 utils 是 append(v), append(u) 然后 set 去重
                # 这里我们单向添加，最后统一去重即可
                graph[int(neighbor)].append(int(i))
    
    # 去重
    final_graph = {k: list(set(v)) for k, v in graph.items()}
    return final_graph

def merge_graphs(output_dir, image_root, mode="book"):
    """
    mode: 'book' (按书籍合并) 或 'global' (全局合并)
    功能：
    1. 遍历 image_root 构建 {章节名: 书籍名} 的映射
    2. 读取 embeddings/ 下所有 .pt
    3. 利用映射关系将章节归类到对应的书籍 (或全局)
    4. 拼接 Tensor，构建大图并保存 Meta
    """
    print(f"Starting merge process in mode: {mode}")
    
    # --- [新增步骤] 构建 章节 -> 书籍 映射 ---
    print(f"Building chapter-to-book mapping from {image_root}...")
    chapter_to_book = {}
    if not os.path.exists(image_root):
        print(f"Error: Image root {image_root} not found.")
        return

    # 遍历 imgs/书籍名/章节名
    for book_name in os.listdir(image_root):
        book_path = os.path.join(image_root, book_name)
        if os.path.isdir(book_path):
            for chapter_name in os.listdir(book_path):
                if os.path.isdir(os.path.join(book_path, chapter_name)):
                    # 记录映射关系
                    chapter_to_book[chapter_name] = book_name
    
    print(f"Mapped {len(chapter_to_book)} chapters to {len(set(chapter_to_book.values()))} books.")

    # --- 开始处理 Embeddings ---
    emb_dir = os.path.join(output_dir, "embeddings")
    if not os.path.exists(emb_dir):
        print("No embeddings found. Run prepare_data first.")
        return

    # 1. 收集所有章节 Embedding 文件
    files = sorted([f for f in os.listdir(emb_dir) if f.endswith(".pt")])
    
    # 2. 分组逻辑
    from collections import defaultdict
    groups = defaultdict(list)
    
    for f in files:
        doc_id = f.replace(".pt", "") # 这里 doc_id 即为 章节名
        
        if mode == "book":
            # [修改] 从映射表中查找书籍名
            group_id = chapter_to_book.get(doc_id)
            if not group_id:
                print(f"Warning: Chapter '{doc_id}' not found in image_root, skipping or assigning to unknown.")
                group_id = "unknown_book"
        else:
            group_id = "global_graph"
            
        groups[group_id].append(doc_id)

    # 3. 遍历分组进行合并
    save_emb_dir = os.path.join(output_dir, f"{mode}_embeddings")
    save_graph_dir = os.path.join(output_dir, f"{mode}_graphs")
    os.makedirs(save_emb_dir, exist_ok=True)
    os.makedirs(save_graph_dir, exist_ok=True)

    for group_id, chapter_ids in tqdm(groups.items(), desc=f"Merging {mode} graphs"):
        # 对章节进行排序，确保拼接顺序正确
        # 尝试解析章节名中的数字进行排序，如果失败则按字母序
        try:
            # 假设章节名格式类似于 "chapter_1", "unit-2" 等，尝试提取末尾数字
            chapter_ids.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else x)
        except:
            chapter_ids.sort()

        merged_embs = []
        metadata = [] # 记录 (chapter_id, start_idx, length)
        current_offset = 0

        # 加载并拼接
        for cid in chapter_ids:
            path = os.path.join(emb_dir, f"{cid}.pt")
            try:
                emb = torch.load(path, map_location="cpu")
            except Exception as e:
                print(f"Error loading {cid}: {e}")
                continue

            # 确保是 [N, 32, 128]
            if len(emb.shape) == 2: 
                 raise ValueError(f"Embedding {cid} shape error. Expected [N, 32, 128]")
            
            num_pages = emb.shape[0]
            merged_embs.append(emb)
            
            metadata.append({
                "doc_id": cid,          # 原始章节名
                "book_id": group_id,    # 书籍名 (新增，方便反查)
                "start": current_offset,
                "length": num_pages
            })
            current_offset += num_pages
        
        if not merged_embs:
            continue

        # 拼接 Tensor
        full_embedding = torch.cat(merged_embs, dim=0)
        
        # 保存合并后的 Embedding
        torch.save(full_embedding, os.path.join(save_emb_dir, f"{group_id}.pt"))
        
        # 保存 Metadata
        with open(os.path.join(save_emb_dir, f"{group_id}_meta.json"), "w", encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        del merged_embs
        import gc
        gc.collect()

        try:
            # 这里的 threshold=0.7 是基于 Cosine Similarity 的
            # 全局图比较大，为了避免边太多，可以适当提高阈值，例如 0.75 或 0.8
            graph = build_graph_gpu_fast(full_embedding, threshold=0.75, k_value=5, device="cuda")
            
            with open(os.path.join(save_graph_dir, f"{group_id}.pkl"), "wb") as f:
                pickle.dump(graph, f)
            print(f"Graph {group_id} built with {len(graph)} nodes.")
            
        except RuntimeError as e:
            print(f"GPU OOM or Error during graph build for {group_id}: {e}")
            print("Falling back to CPU (This will be very slow)...")
            # Fallback (仅当 GPU 显存实在不够时)
            doc_emb_input = full_embedding.float().numpy()
            graph = construct_page_graph(doc_emb_input, threshold=0.7, k_value=5)
            with open(os.path.join(save_graph_dir, f"{group_id}.pkl"), "wb") as f:
                pickle.dump(graph, f)

        # # 构建并保存大图
        # # 注意转为 numpy 供 datautil 使用
        # doc_emb_input = full_embedding.float().numpy()
        # # 书籍/全局层面图可能较大，threshold=0.7 是官方默认值，可根据密度调整
        # graph = construct_page_graph(doc_emb_input, threshold=0.7, k_value=5)
        
        # with open(os.path.join(save_graph_dir, f"{group_id}.pkl"), "wb") as f:
        #     pickle.dump(graph, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/helq/doc_re/m3bookvqa/m3bookvqa/data.jsonl")
    parser.add_argument("--image_root", type=str, default="/data/helq/doc_re/m3bookvqa/imgs")
    parser.add_argument("--output_dir", type=str, default="/data/helq/doc_re/m3bookvqa/molorag")
    parser.add_argument("--mode", type=str, default="merge_global", choices=["prepare", "merge_book", "merge_global"])
    args = parser.parse_args()

    # 根据 mode 执行不同逻辑
    if args.mode == "prepare":
        # 执行原始的 Embedding 生成和单章节构图
        prepare_data(args.data_path, args.image_root, args.output_dir)
    
    elif args.mode == "merge_book":
        # 执行书籍层面的合并
        # 需要 import re 用于排序逻辑
        import re 
        merge_graphs(args.output_dir, args.image_root, mode="book")
        
    elif args.mode == "merge_global":
        # 执行全局层面的合并
        import re
        merge_graphs(args.output_dir, args.image_root, mode="global")