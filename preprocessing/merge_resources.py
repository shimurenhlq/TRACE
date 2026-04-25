"""
Merge chapter-level resources into book-level or global-level resources.
Creates merged embeddings, graphs, and metadata for multi-chapter retrieval.
"""

import os
import json
import torch
import pickle
import re
import argparse
from tqdm import tqdm
from collections import defaultdict


def build_graph_gpu_fast(embeddings, threshold=0.7, k_value=5, device="cuda"):
    """
    Build graph using GPU-accelerated cosine similarity computation.

    Args:
        embeddings: Tensor of shape [N, 32, 128]
        threshold: Similarity threshold for edge creation
        k_value: Maximum number of neighbors per node
        device: Device to use for computation

    Returns:
        Dictionary representing adjacency list
    """
    print(f"Building graph on {device} for {embeddings.shape[0]} pages...")
    n_pages = embeddings.shape[0]

    # Mean pooling: [N, 32, 128] -> [N, 128]
    embs_mean = embeddings.mean(dim=1).to(device, dtype=torch.float32)

    # L2 normalize
    embs_mean = torch.nn.functional.normalize(embs_mean, p=2, dim=1)

    # Compute cosine similarity matrix
    sim_matrix = torch.matmul(embs_mean, embs_mean.T)

    # Get top-k for each page
    topk_values, topk_indices = torch.topk(sim_matrix, k=k_value, dim=1)

    topk_values = topk_values.cpu().numpy()
    topk_indices = topk_indices.cpu().numpy()

    graph = defaultdict(list)

    for i in tqdm(range(n_pages), desc="Building adjacency list"):
        for k in range(k_value):
            score = topk_values[i, k]
            neighbor = topk_indices[i, k]

            if neighbor == i:  # Skip self-loops
                continue

            if score >= threshold:
                graph[int(i)].append(int(neighbor))
                graph[int(neighbor)].append(int(i))  # Undirected graph

    # Remove duplicates
    final_graph = {k: list(set(v)) for k, v in graph.items()}
    return final_graph


def merge_resources(input_dir: str, image_root: str, mode: str = "book",
                   threshold: float = 0.75, k_value: int = 5, device: str = "cuda"):
    """
    Merge chapter-level resources into book-level or global-level.

    Args:
        input_dir: Directory containing chapter-level embeddings and graphs
        image_root: Root directory of images (to determine book structure)
        mode: "book" or "global"
        threshold: Graph similarity threshold
        k_value: Maximum neighbors per node
        device: Device for graph construction
    """
    print(f"Starting merge in {mode} mode...")

    # Build chapter-to-book mapping
    print(f"Building chapter-to-book mapping from {image_root}...")
    chapter_to_book = {}

    if not os.path.exists(image_root):
        raise FileNotFoundError(f"Image root not found: {image_root}")

    for book_name in os.listdir(image_root):
        book_path = os.path.join(image_root, book_name)
        if os.path.isdir(book_path):
            for chapter_name in os.listdir(book_path):
                if os.path.isdir(os.path.join(book_path, chapter_name)):
                    chapter_to_book[chapter_name] = book_name

    print(f"Mapped {len(chapter_to_book)} chapters to {len(set(chapter_to_book.values()))} books")

    # Collect embedding files
    emb_dir = os.path.join(input_dir, "embeddings")
    if not os.path.exists(emb_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")

    files = sorted([f for f in os.listdir(emb_dir) if f.endswith(".pt")])

    # Group chapters
    groups = defaultdict(list)

    for f in files:
        doc_id = f.replace(".pt", "")

        if mode == "book":
            group_id = chapter_to_book.get(doc_id, "unknown_book")
        else:  # global
            group_id = "global_graph"

        groups[group_id].append(doc_id)

    # Create output directories
    save_emb_dir = os.path.join(input_dir, f"{mode}_embeddings")
    save_graph_dir = os.path.join(input_dir, f"{mode}_graphs")
    os.makedirs(save_emb_dir, exist_ok=True)
    os.makedirs(save_graph_dir, exist_ok=True)

    # Process each group
    for group_id, chapter_ids in tqdm(groups.items(), desc=f"Merging {mode} groups"):
        # Sort chapters by numeric suffix if possible
        try:
            chapter_ids.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else x)
        except:
            chapter_ids.sort()

        merged_embs = []
        metadata = []
        current_offset = 0

        # Load and concatenate embeddings
        for cid in chapter_ids:
            path = os.path.join(emb_dir, f"{cid}.pt")
            try:
                emb = torch.load(path, map_location="cpu")
            except Exception as e:
                print(f"Error loading {cid}: {e}")
                continue

            if len(emb.shape) != 3:
                print(f"Warning: Unexpected shape for {cid}: {emb.shape}")
                continue

            num_pages = emb.shape[0]
            merged_embs.append(emb)

            metadata.append({
                "doc_id": cid,
                "book_id": group_id,
                "start": current_offset,
                "length": num_pages
            })
            current_offset += num_pages

        if not merged_embs:
            print(f"Warning: No valid embeddings for group {group_id}")
            continue

        # Concatenate embeddings
        full_embedding = torch.cat(merged_embs, dim=0)

        # Save merged embeddings
        torch.save(full_embedding, os.path.join(save_emb_dir, f"{group_id}.pt"))

        # Save metadata
        with open(os.path.join(save_emb_dir, f"{group_id}_meta.json"), "w", encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Build and save graph
        try:
            graph = build_graph_gpu_fast(full_embedding, threshold=threshold, k_value=k_value, device=device)

            with open(os.path.join(save_graph_dir, f"{group_id}.pkl"), "wb") as f:
                pickle.dump(graph, f)

            print(f"Graph for {group_id}: {len(graph)} nodes, {sum(len(v) for v in graph.values())} edges")
        except Exception as e:
            print(f"Error building graph for {group_id}: {e}")

    print(f"Merge complete. Output saved to {save_emb_dir} and {save_graph_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge chapter resources into book/global level")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing chapter-level embeddings and graphs")
    parser.add_argument("--image_root", type=str, required=True,
                       help="Root directory of images")
    parser.add_argument("--mode", type=str, choices=["book", "global"], required=True,
                       help="Merge mode: book or global")
    parser.add_argument("--threshold", type=float, default=0.75,
                       help="Graph similarity threshold")
    parser.add_argument("--k_value", type=int, default=5,
                       help="Maximum neighbors per node")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for graph construction")

    args = parser.parse_args()

    merge_resources(
        input_dir=args.input_dir,
        image_root=args.image_root,
        mode=args.mode,
        threshold=args.threshold,
        k_value=args.k_value,
        device=args.device
    )


if __name__ == "__main__":
    main()
