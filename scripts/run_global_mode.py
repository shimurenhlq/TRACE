"""
Run TRACE in global mode (maximum coverage, cross-book retrieval).
"""

import os
import sys
import json
import asyncio
import argparse
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.environment import BookEnvironment
from src.agents import AgenticSystem


async def main():
    parser = argparse.ArgumentParser(description="Run TRACE in global mode")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data.jsonl")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory of images")
    parser.add_argument("--asset_dir", type=str, required=True, help="Directory with embeddings and graphs")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    raw_log_dir = os.path.join(os.path.dirname(args.output_file), "raw_logs_global")
    os.makedirs(raw_log_dir, exist_ok=True)

    # Load configuration
    config = Config(args.config)

    # Initialize environment and system
    print("Initializing BookEnvironment...")
    book_env = BookEnvironment(args.asset_dir, args.image_root, config)

    print("Initializing AgenticSystem...")
    system = AgenticSystem(config)

    # Load data
    print(f"Loading data from {args.data_path}...")
    data = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Slice data if specified
    end_idx = args.end_idx if args.end_idx else len(data)
    data = data[args.start_idx:end_idx]
    print(f"Processing {len(data)} samples (index {args.start_idx} to {end_idx})")

    # Load global resources (only once)
    print("Loading global resources (this may take a while)...")
    try:
        book_env.load_resources(group_id="global_graph", mode="global")
    except Exception as e:
        print(f"Failed to load global resources: {e}")
        return

    # Process samples
    results = []

    for sample in tqdm(data, desc="Processing samples"):
        doc_id = sample['id'].rsplit('-', 1)[0]

        try:
            # Solve question
            result_pkg = await system.solve(sample['question'], sample['options'], book_env)

            # Save detailed logs
            log_path = os.path.join(raw_log_dir, f"{sample['id']}.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(result_pkg["logs"])

            # Convert indices to UIDs
            retrieved_uids = [book_env.index_to_uid(idx) for idx in result_pkg["retrieved_indices"]]
            gt_uids = [f"{doc_id}_page{p}" for p in sample['page_numbers']]

            # Create result entry
            res_entry = {
                "id": sample['id'],
                "question": sample['question'],
                "ground_truth_answer": sample['answer'],
                "predicted_answer": result_pkg["pred_answer"],
                "ground_truth_pages": gt_uids,
                "retrieved_pages": retrieved_uids,
                "options": sample['options']
            }

            results.append(res_entry)

            print(f"  {sample['id']}: GT={sample['answer']}, Pred={result_pkg['pred_answer']}")

        except Exception as e:
            print(f"Error processing {sample['id']}: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                "id": sample['id'],
                "question": sample['question'],
                "ground_truth_answer": sample['answer'],
                "predicted_answer": "ERROR",
                "ground_truth_pages": [],
                "retrieved_pages": [],
                "options": sample['options'],
                "error": str(e)
            })

        # Save incrementally
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete. Results saved to {args.output_file}")

    # Calculate accuracy
    correct = sum(1 for r in results if r['predicted_answer'] == r['ground_truth_answer'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
