import os
import json
import asyncio
import argparse
from tqdm import tqdm
from collections import defaultdict

# 导入共享组件
from agent_shared import BookEnvironment, AgenticSystem

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/helq/doc_re/m3bookvqa/m3bookvqa/data.jsonl")
    parser.add_argument("--image_root", type=str, default="/data/helq/doc_re/m3bookvqa/imgs")
    parser.add_argument("--molorag_dir", type=str, default="/data/helq/doc_re/m3bookvqa/molorag")
    parser.add_argument("--output_file", type=str, default="/data/helq/doc_re/m3bookvqa/agenticRAG/results_book_pangu/results_book.json")
    args = parser.parse_args()

    raw_log_dir = os.path.join(os.path.dirname(args.output_file), "raw_book")
    os.makedirs(raw_log_dir, exist_ok=True)

    # 1. 初始化
    book_env = BookEnvironment(args.molorag_dir, args.image_root)
    system = AgenticSystem()

    # 2. 读取数据并按书籍分组
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
            
    # 构建 章节 -> 书籍 的映射
    # 方法：doc_id 通常是章节名。我们需要知道它属于哪本书。
    # 利用 BookEnvironment 的 chapter_index: {章节名: full_path}
    # full_path 形如 .../imgs/书名/章节名
    doc_to_book = {}
    for doc_id, path in book_env.chapter_index.items():
        parent_dir = os.path.dirname(path) # .../imgs/书名
        book_name = os.path.basename(parent_dir)
        doc_to_book[doc_id] = book_name

    # 分组
    book_groups = defaultdict(list)
    for sample in data:
        doc_id = sample['id'].rsplit('-', 1)[0]
        book_name = doc_to_book.get(doc_id)
        if book_name:
            book_groups[book_name].append(sample)
        else:
            print(f"Warning: Book not found for doc {doc_id}")

    # 3. 按书执行
    results = []
    print(f"Processing {len(book_groups)} books...")

    for book_name, samples in tqdm(book_groups.items(), desc="Books"):
        # 加载该书资源
        try:
            book_env.load_resources(group_id=book_name, mode="book")
        except Exception as e:
            print(f"Skipping book {book_name}: {e}")
            continue
            
        for sample in tqdm(samples, desc=f"Samples in {book_name}", leave=False):
            doc_id = sample['id'].rsplit('-', 1)[0]
            try:
                res_pkg = await system.solve(sample['question'], sample['options'], book_env)
                
                # 保存日志
                with open(os.path.join(raw_log_dir, f"{sample['id']}.txt"), "w") as f:
                    f.write(res_pkg["logs"])
                
                # 转换结果 ID
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
            if len(results) % 1 == 0:
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())