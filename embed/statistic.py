import os
import random
import torch
import pickle
import numpy as np

def analyze_data_stats(output_dir):
    """
    分析指定目录下的 embeddings (.pt) 和 graphs (.pkl) 数据统计信息
    """
    emb_dir = os.path.join(output_dir, "embeddings")
    graph_dir = os.path.join(output_dir, "graphs")

    print(f"=== 开始分析数据: {output_dir} ===\n")

    # ---------------------------------------------------------
    # 1. Embeddings 分析 (随机抽样 5 个)
    # ---------------------------------------------------------
    print(f"--- [1/2] Embedding 文件分析 ({emb_dir}) ---")
    if not os.path.exists(emb_dir):
        print(f"错误: 目录不存在 {emb_dir}")
    else:
        emb_files = [f for f in os.listdir(emb_dir) if f.endswith('.pt')]
        count = len(emb_files)
        print(f"共发现 {count} 个 Embedding 文件。")

        if count > 0:
            # 随机抽取 5 个，不足 5 个则全取
            sample_size = min(5, count)
            selected_files = random.sample(emb_files, sample_size)
            
            print(f"随机读取 {sample_size} 个文件查看 Shape:")
            for f_name in selected_files:
                file_path = os.path.join(emb_dir, f_name)
                try:
                    # map_location='cpu' 防止在无 GPU 环境下报错
                    data = torch.load(file_path, map_location='cpu')
                    # 获取 shape
                    shape_str = str(list(data.shape))
                    print(f"  - 文件: {f_name:<40} Shape: {shape_str}")
                except Exception as e:
                    print(f"  - 读取失败 {f_name}: {e}")
        print("")

    # ---------------------------------------------------------
    # 2. Graphs 分析 (全量统计)
    # ---------------------------------------------------------
    print(f"--- [2/2] Graph 文件统计 ({graph_dir}) ---")
    if not os.path.exists(graph_dir):
        print(f"错误: 目录不存在 {graph_dir}")
    else:
        graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.pkl')]
        graph_count = len(graph_files)
        
        if graph_count == 0:
            print("未发现 .pkl 图文件。")
        else:
            print(f"正在统计 {graph_count} 个图文件，请稍候...")
            
            total_nodes = 0
            total_edges = 0
            valid_graphs = 0
            
            # 用于记录最大/最小值
            max_nodes = 0
            max_edges = 0
            
            for f_name in graph_files:
                file_path = os.path.join(graph_dir, f_name)
                try:
                    with open(file_path, 'rb') as f:
                        graph = pickle.load(f)
                    
                    # 兼容性处理：通常 graph 是一个邻接表字典 {node_idx: [neighbor_idxs...]}
                    n_nodes = 0
                    n_edges = 0
                    
                    if isinstance(graph, dict):
                        n_nodes = len(graph)
                        # 统计所有值的列表长度之和作为边数 (有向边总数)
                        n_edges = sum(len(neighbors) for neighbors in graph.values())
                    # 如果存储的是 NetworkX 对象
                    elif hasattr(graph, 'number_of_nodes') and hasattr(graph, 'number_of_edges'):
                        n_nodes = graph.number_of_nodes()
                        n_edges = graph.number_of_edges()
                    
                    total_nodes += n_nodes
                    total_edges += n_edges
                    valid_graphs += 1
                    
                    max_nodes = max(max_nodes, n_nodes)
                    max_edges = max(max_edges, n_edges)
                    
                except Exception as e:
                    print(f"  - 读取图文件失败 {f_name}: {e}")

            if valid_graphs > 0:
                avg_nodes = total_nodes / valid_graphs
                avg_edges = total_edges / valid_graphs
                # 平均度 = 平均每个节点连接了多少其他节点
                avg_degree = avg_edges / avg_nodes if avg_nodes > 0 else 0
                
                print("-" * 40)
                print(f"统计结果 (基于 {valid_graphs} 个有效文件):")
                print(f"  - 平均节点数 (Pages): {avg_nodes:.2f} (Max: {max_nodes})")
                print(f"  - 平均边数 (Links):   {avg_edges:.2f} (Max: {max_edges})")
                print(f"  - 平均度 (Edges/Node): {avg_degree:.2f}")
                print("-" * 40)
                
                # --- 阈值合理性建议 ---
                print("阈值合理性检查:")
                if avg_degree == 0:
                    print("  [警报] 图是孤立的（没有边）。\n  -> 建议：大幅**降低** threshold (当前0.7可能太高)。")
                elif avg_degree < 2:
                    print(f"  [提示] 图非常稀疏 (平均每个页面仅连接 {avg_degree:.1f} 个页面)。\n  -> 建议：适当**降低** threshold 以增加连通性。")
                elif avg_degree > 15:
                    print(f"  [警报] 图过于稠密 (平均每个页面连接 {avg_degree:.1f} 个页面)。\n  -> 建议：**提高** threshold，否则检索会引入过多噪声。")
                else:
                    print(f"  [OK] 图的稀疏程度看起来比较合理 (平均度 {avg_degree:.1f})。")
            else:
                print("没有有效读取到图数据。")

if __name__ == "__main__":
    # 根据你之前的代码设置路径
    OUTPUT_DIR = "/data/helq/doc_re/m3bookvqa/molorag"
    
    analyze_data_stats(OUTPUT_DIR)