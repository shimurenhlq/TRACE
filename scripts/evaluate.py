"""
Evaluate TRACE results and compute metrics.
"""

import json
import argparse
from collections import defaultdict


def evaluate_results(result_file: str):
    """
    Evaluate results and compute accuracy metrics.

    Args:
        result_file: Path to results JSON file
    """
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    total = len(results)
    correct = 0
    errors = 0

    # Per-topic accuracy
    topic_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for r in results:
        if r['predicted_answer'] == 'ERROR':
            errors += 1
            continue

        if r['predicted_answer'] == r['ground_truth_answer']:
            correct += 1

        # Track by topic if available
        topic = r.get('topic', 'unknown')
        topic_stats[topic]['total'] += 1
        if r['predicted_answer'] == r['ground_truth_answer']:
            topic_stats[topic]['correct'] += 1

    # Overall accuracy
    valid_total = total - errors
    accuracy = correct / valid_total if valid_total > 0 else 0

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Errors: {errors}")
    print(f"Valid samples: {valid_total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{valid_total})")
    print()

    # Per-topic breakdown
    if len(topic_stats) > 1:
        print("Per-topic accuracy:")
        print("-" * 60)
        for topic, stats in sorted(topic_stats.items()):
            topic_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {topic:20s}: {topic_acc:.2%} ({stats['correct']}/{stats['total']})")
        print()

    # Retrieval statistics
    print("Retrieval statistics:")
    print("-" * 60)

    total_retrieved = sum(len(r.get('retrieved_pages', [])) for r in results if r['predicted_answer'] != 'ERROR')
    avg_retrieved = total_retrieved / valid_total if valid_total > 0 else 0
    print(f"  Average pages retrieved: {avg_retrieved:.2f}")

    # Compute recall (how many ground truth pages were retrieved)
    recalls = []
    for r in results:
        if r['predicted_answer'] == 'ERROR':
            continue
        gt_pages = set(r.get('ground_truth_pages', []))
        ret_pages = set(r.get('retrieved_pages', []))
        if len(gt_pages) > 0:
            recall = len(gt_pages & ret_pages) / len(gt_pages)
            recalls.append(recall)

    if recalls:
        avg_recall = sum(recalls) / len(recalls)
        print(f"  Average page recall: {avg_recall:.2%}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRACE results")
    parser.add_argument("--result_file", type=str, required=True, help="Path to results JSON file")

    args = parser.parse_args()

    evaluate_results(args.result_file)


if __name__ == "__main__":
    main()
