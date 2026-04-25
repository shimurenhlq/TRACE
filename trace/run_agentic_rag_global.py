import os
import json
import asyncio
import argparse
from tqdm import tqdm

# 导入共享组件
from agent_shared import BookEnvironment, AgenticSystem

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/helq/doc_re/m3bookvqa/m3bookvqa/data.jsonl")
    parser.add_argument("--image_root", type=str, default="/data/helq/doc_re/m3bookvqa/imgs")
    parser.add_argument("--molorag_dir", type=str, default="/data/helq/doc_re/m3bookvqa/molorag")
    parser.add_argument("--output_file", type=str, default="/data/helq/doc_re/m3bookvqa/agenticRAG/results_global_pangu/results_global.json")
    args = parser.parse_args()

    raw_log_dir = os.path.join(os.path.dirname(args.output_file), "raw_global")
    os.makedirs(raw_log_dir, exist_ok=True)

    # 1. 初始化
    book_env = BookEnvironment(args.molorag_dir, args.image_root)
    system = AgenticSystem()

    # 2. 读取数据
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))

    # 3. 加载全局资源 (只加载一次)
    print("Loading GLOBAL resources (this may take a while)...")
    try:
        book_env.load_resources(group_id="global_graph", mode="global")
    except Exception as e:
        print(f"Failed to load global resources: {e}")
        return

    # 4. 全局循环
    results = []
    print(f"Processing {len(data)} samples in GLOBAL mode...")

    for sample in tqdm(data):
        doc_id = sample['id'].rsplit('-', 1)[0]
        try:
            # 核心区别：此时 book_env 内部已经是全量数据，solve 在全量数据上运行
            res_pkg = await system.solve(sample['question'], sample['options'], book_env)
            
            # 保存日志
            with open(os.path.join(raw_log_dir, f"{sample['id']}.txt"), "w") as f:
                f.write(res_pkg["logs"])
            
            # 转换结果 ID (index_to_uid 会自动处理全局索引到具体书/章节的映射)
            ret_indices = res_pkg["retrieved_indices"]
            ret_uids = [book_env.index_to_uid(idx) for idx in ret_indices]
            
            res_entry = {
                "id": sample['id'],
                "question": sample['question'],
                "ground_truth_answer": sample['answer'],
                "predicted_answer": res_pkg["pred_answer"],
                "ground_truth_pages": [f"{doc_id}_page{p}" for p in sample['page_numbers']],
                "retrieved_pages": ret_uids,
                "options": sample['options']
            }
            results.append(res_entry)
            
        except Exception as e:
            print(f"Error sample {sample['id']}: {e}")

        # 实时保存
        if len(results) % 5 == 0:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())