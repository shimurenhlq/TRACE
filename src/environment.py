"""
BookEnvironment: Manages document resources, embeddings, and retrieval.
Supports chapter-level, book-level, and global-level retrieval modes.
"""

import os
import json
import pickle
import torch
from typing import List, Dict, Optional, Tuple
from bisect import bisect_right

from colpali_engine.models import ColPali, ColPaliProcessor
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .config import Config


class BookEnvironment:
    """
    Environment class that maintains document state, embeddings, graphs, and retrieval models.
    Supports three operational modes: chapter, book, and global.
    """

    def __init__(self, asset_dir: str, image_root: str, config: Config):
        """
        Initialize the BookEnvironment.

        Args:
            asset_dir: Directory containing preprocessed embeddings and graphs
            image_root: Root directory containing document images
            config: Configuration object with model and retrieval settings
        """
        self.asset_dir = asset_dir
        self.image_root = image_root
        self.config = config
        self.device = config.colpali_device

        print(f"Loading ColPali model on {self.device}...")
        self.colpali = ColPali.from_pretrained(
            config.colpali_model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
            local_files_only=True
        ).eval()
        self.colpali_processor = ColPaliProcessor.from_pretrained(
            config.colpali_model_path,
            local_files_only=True
        )

        # Initialize vision client for Navigator
        self.vision_client = OpenAIChatCompletionClient(
            api_key=config.navigator.api_key,
            base_url=config.navigator.base_url,
            model=config.navigator.model,
            model_info=config.navigator.model_info or {}
        )

        # Build chapter index: chapter_name -> image_directory
        self.chapter_index = self._build_chapter_index()

        # Resource cache
        self.current_group_id = None
        self.current_embeddings = None  # Tensor of embeddings
        self.current_graph = None  # Graph adjacency dict
        self.meta_segments = []  # Metadata for index mapping
        self.meta_starts = []  # For binary search
        self._dir_cache = {}  # Cache for image file lists

    def _build_chapter_index(self) -> Dict[str, str]:
        """
        Build mapping from chapter name to image directory path.

        Returns:
            Dictionary mapping chapter_name -> full_path_to_images
        """
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

    def load_document(self, doc_id: str):
        """
        Load a single chapter's resources (chapter mode).

        Args:
            doc_id: Chapter identifier
        """
        if self.current_group_id == doc_id:
            return

        print(f"Loading chapter: {doc_id}")
        self.current_group_id = doc_id

        emb_path = os.path.join(self.asset_dir, "embeddings", f"{doc_id}.pt")
        graph_path = os.path.join(self.asset_dir, "graphs", f"{doc_id}.pkl")

        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embeddings not found: {emb_path}")

        self.current_embeddings = torch.load(emb_path, map_location="cpu").to(torch.bfloat16)

        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                self.current_graph = pickle.load(f)
        else:
            self.current_graph = {}

        # For chapter mode, create simple metadata
        self.meta_segments = [{
            'doc_id': doc_id,
            'start': 0,
            'length': len(self.current_embeddings)
        }]
        self.meta_starts = [0]

    def load_resources(self, group_id: str, mode: str = "book"):
        """
        Load resources for book-level or global-level retrieval.

        Args:
            group_id: Book name (for book mode) or "global_graph" (for global mode)
            mode: "book" or "global"
        """
        if self.current_group_id == group_id:
            return

        print(f"Loading resources for {mode} mode: {group_id}")
        self.current_group_id = group_id

        prefix = "book" if mode == "book" else "global"

        # Load embeddings
        emb_path = os.path.join(self.asset_dir, f"{prefix}_embeddings", f"{group_id}.pt")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embeddings not found: {emb_path}")
        self.current_embeddings = torch.load(emb_path, map_location="cpu").to(torch.bfloat16)

        # Load graph
        graph_path = os.path.join(self.asset_dir, f"{prefix}_graphs", f"{group_id}.pkl")
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                self.current_graph = pickle.load(f)
        else:
            self.current_graph = {}

        # Load metadata
        meta_path = os.path.join(self.asset_dir, f"{prefix}_embeddings", f"{group_id}_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        with open(meta_path, 'r') as f:
            self.meta_segments = json.load(f)

        self.meta_starts = [seg['start'] for seg in self.meta_segments]

    def _resolve_global_idx(self, global_idx: int) -> Tuple[Optional[str], int]:
        """
        Map global index to (doc_id, local_page_idx).

        Args:
            global_idx: Global page index

        Returns:
            Tuple of (doc_id, local_page_idx) or (None, -1) if invalid
        """
        seg_idx = bisect_right(self.meta_starts, global_idx) - 1
        if seg_idx < 0 or seg_idx >= len(self.meta_segments):
            return None, -1

        segment = self.meta_segments[seg_idx]
        if not (segment['start'] <= global_idx < segment['start'] + segment['length']):
            return None, -1

        local_page_idx = global_idx - segment['start']
        return segment['doc_id'], local_page_idx

    def get_image_path(self, page_idx: int) -> Optional[str]:
        """
        Get image file path for a given page index.

        Args:
            page_idx: Page index (0-based, can be global or local depending on mode)

        Returns:
            Full path to image file, or None if not found
        """
        # For chapter mode
        if len(self.meta_segments) == 1 and self.meta_segments[0]['start'] == 0:
            doc_id = self.meta_segments[0]['doc_id']
            local_idx = page_idx
        else:
            # For book/global mode
            doc_id, local_idx = self._resolve_global_idx(page_idx)
            if not doc_id:
                return None

        chapter_dir = self.chapter_index.get(doc_id)
        if not chapter_dir:
            return None

        # Cache directory listing
        if doc_id not in self._dir_cache:
            files = sorted(
                [f for f in os.listdir(chapter_dir) if f.endswith(('.png', '.jpg'))],
                key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else 0
            )
            self._dir_cache[doc_id] = [os.path.join(chapter_dir, f) for f in files]

        files = self._dir_cache[doc_id]
        if 0 <= local_idx < len(files):
            return files[local_idx]
        return None

    def index_to_uid(self, page_idx: int) -> str:
        """
        Convert page index to human-readable UID (e.g., "chapter1_page5").

        Args:
            page_idx: Page index (0-based)

        Returns:
            UID string
        """
        if len(self.meta_segments) == 1 and self.meta_segments[0]['start'] == 0:
            doc_id = self.meta_segments[0]['doc_id']
            local_idx = page_idx
        else:
            doc_id, local_idx = self._resolve_global_idx(page_idx)
            if not doc_id:
                return f"unknown_{page_idx}"

        return f"{doc_id}_page{local_idx + 1}"

    def retrieve_page_scores(self, query: str) -> torch.Tensor:
        """
        Calculate similarity scores for all pages given a query.

        Args:
            query: Search query string

        Returns:
            Tensor of scores [num_pages]
        """
        if self.current_embeddings is None:
            return torch.tensor([])

        with torch.no_grad():
            batch_query = self.colpali_processor.process_queries([query]).to(self.device)
            query_emb = self.colpali(**batch_query)

            doc_emb_device = self.current_embeddings.to(self.device)
            scores = self.colpali_processor.score_multi_vector(query_emb, doc_emb_device)

        return scores[0].cpu()

    def retrieve_initial_pages(self, query: str) -> List[int]:
        """
        Retrieve all page indices sorted by similarity (ascending order).

        Args:
            query: Search query string

        Returns:
            List of page indices sorted by score (low to high)
        """
        scores = self.retrieve_page_scores(query)
        if len(scores) == 0:
            return []
        sorted_indices = scores.argsort(descending=False)
        return sorted_indices.tolist()

    def get_semantic_neighbors(self, page_idx: int, k: int = 3) -> List[int]:
        """
        Get semantic neighbors from MoLoRAG graph.

        Args:
            page_idx: Page index
            k: Number of neighbors to return

        Returns:
            List of neighbor page indices
        """
        if not self.current_graph or page_idx not in self.current_graph:
            return []

        neighbors = list(self.current_graph[page_idx])
        return neighbors[:k]
