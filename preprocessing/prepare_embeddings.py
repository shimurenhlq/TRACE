"""
Generate ColPali embeddings and build MoLoRAG graphs for each chapter.
This script processes individual chapters and creates chapter-level resources.
"""

import os
import json
import torch
import pickle
import argparse
from tqdm import tqdm
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor


def construct_page_graph_simple(embeddings, threshold=0.7, k_value=5):
    """
    Construct page graph using cosine similarity.

    Args:
        embeddings: numpy array of shape [N, 32, 128] (ColPali multi-vector embeddings)
        threshold: Similarity threshold for edge creation
        k_value: Maximum number of neighbors per node

    Returns:
        Dictionary representing adjacency list {page_idx: [neighbor_indices]}
    """
    import numpy as np
    from collections import defaultdict

    n_pages = embeddings.shape[0]
    graph = defaultdict(list)

    # Mean pooling: [N, 32, 128] -> [N, 128]
    emb_mean = embeddings.mean(axis=1)

    # Normalize for cosine similarity
    emb_norm = emb_mean / (np.linalg.norm(emb_mean, axis=1, keepdims=True) + 1e-8)

    # Compute similarity matrix
    sim_matrix = np.dot(emb_norm, emb_norm.T)

    # Build graph
    for i in range(n_pages):
        # Get top-k similar pages (excluding self)
        similarities = sim_matrix[i]
        top_indices = np.argsort(similarities)[::-1]

        count = 0
        for j in top_indices:
            if j == i:
                continue
            if similarities[j] >= threshold and count < k_value:
                graph[i].append(int(j))
                count += 1
            if count >= k_value:
                break

    return dict(graph)


def prepare_embeddings(data_path: str, image_root: str, output_dir: str,
                      colpali_model_path: str, device: str = "cuda", batch_size: int = 1):
    """
    Generate ColPali embeddings and MoLoRAG graphs for each chapter.

    Args:
        data_path: Path to data.jsonl file
        image_root: Root directory containing book/chapter/image structure
        output_dir: Output directory for embeddings and graphs
        colpali_model_path: Path to ColPali model
        device: Device to use for computation
        batch_size: Batch size for embedding generation
    """
    os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)

    # Load ColPali model
    print(f"Loading ColPali model from {colpali_model_path}...")
    model = ColPali.from_pretrained(
        colpali_model_path,
        dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True
    ).eval()
    processor = ColPaliProcessor.from_pretrained(colpali_model_path, local_files_only=True)

    # Load data
    print(f"Loading data from {data_path}...")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Build chapter index
    print("Building chapter index...")
    chapter_to_path = {}
    if not os.path.exists(image_root):
        raise FileNotFoundError(f"Image root not found: {image_root}")

    for book_name in os.listdir(image_root):
        book_path = os.path.join(image_root, book_name)
        if os.path.isdir(book_path):
            for chapter_name in os.listdir(book_path):
                chapter_path = os.path.join(book_path, chapter_name)
                if os.path.isdir(chapter_path):
                    chapter_to_path[chapter_name] = chapter_path

    print(f"Found {len(chapter_to_path)} chapters")

    # Process each unique chapter
    processed_docs = set()

    print("Processing chapters...")
    for sample in tqdm(data):
        # Extract chapter name from sample ID (format: "chapter_name-question_num")
        doc_id = sample['id'].rsplit('-', 1)[0]

        if doc_id in processed_docs:
            continue

        doc_img_dir = chapter_to_path.get(doc_id)
        if not doc_img_dir:
            raise ValueError(f"Chapter '{doc_id}' not found in image_root")

        if not os.path.exists(doc_img_dir):
            raise FileNotFoundError(f"Chapter directory not found: {doc_img_dir}")

        # Get image files
        image_files = sorted(
            [f for f in os.listdir(doc_img_dir) if f.endswith(('.png', '.jpg'))],
            key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else 0
        )
        image_paths = [os.path.join(doc_img_dir, f) for f in image_files]

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {doc_img_dir}")

        # Generate embeddings
        emb_path = os.path.join(output_dir, "embeddings", f"{doc_id}.pt")

        if not os.path.exists(emb_path):
            print(f"  Generating embeddings for {doc_id} ({len(image_paths)} pages)...")
            loaded_images = [Image.open(p).convert("RGB") for p in image_paths]

            all_embs = []
            with torch.no_grad():
                for i in range(0, len(loaded_images), batch_size):
                    batch_imgs = loaded_images[i:i + batch_size]
                    batch_inputs = processor.process_images(batch_imgs)
                    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                    embs = model(**batch_inputs)
                    all_embs.append(embs.cpu())

            doc_embedding = torch.cat(all_embs, dim=0)
            torch.save(doc_embedding, emb_path)
        else:
            doc_embedding = torch.load(emb_path, map_location="cpu")

        # Build graph
        graph_path = os.path.join(output_dir, "graphs", f"{doc_id}.pkl")

        if not os.path.exists(graph_path):
            print(f"  Building graph for {doc_id}...")
            doc_emb_numpy = doc_embedding.float().numpy()
            graph = construct_page_graph_simple(doc_emb_numpy, threshold=0.7, k_value=5)

            with open(graph_path, 'wb') as f:
                pickle.dump(graph, f)

        processed_docs.add(doc_id)

    print(f"Processing complete. Generated resources for {len(processed_docs)} chapters.")


def main():
    parser = argparse.ArgumentParser(description="Generate ColPali embeddings and MoLoRAG graphs")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data.jsonl")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--colpali_model_path", type=str, default="vidore/colpali-v1.2",
                       help="Path to ColPali model")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for embedding generation")

    args = parser.parse_args()

    prepare_embeddings(
        data_path=args.data_path,
        image_root=args.image_root,
        output_dir=args.output_dir,
        colpali_model_path=args.colpali_model_path,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
