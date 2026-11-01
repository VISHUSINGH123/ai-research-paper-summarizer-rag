"""
FAISS Vector Store wrapper for high-throughput similarity indexing.
"""

import os
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class FAISSStore:
    """FAISS-based dense vector index manager."""

    def __init__(self, embeddings: Embeddings, index_dir: str = "data/cache/faiss_indices"):
        self.embeddings = embeddings
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.db: Optional[FAISS] = None
        self.indexed_count: int = 0
        self.collection_name: str = "default"

    def build_index(self, documents: List[Document], collection_name: str = "default") -> Dict[str, Any]:
        """
        Build a new in-memory FAISS index from documents.
        """
        start_time = time.perf_counter()
        logger.info(f"Building FAISS index '{collection_name}' with {len(documents)} chunks...")

        self.collection_name = collection_name
        self.db = FAISS.from_documents(documents, self.embeddings)
        self.indexed_count = len(documents)
        elapsed_sec = time.perf_counter() - start_time

        logger.info(f"FAISS index built in {elapsed_sec:.3f}s for {len(documents)} items.")
        return {
            "engine": "FAISS",
            "collection_name": collection_name,
            "count": len(documents),
            "indexing_time_sec": elapsed_sec,
        }

    def add_documents(self, documents: List[Document]) -> int:
        """Add additional documents to existing FAISS index."""
        if self.db is None:
            self.build_index(documents)
            return len(documents)
        self.db.add_documents(documents)
        self.indexed_count += len(documents)
        return len(documents)

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search with normalized similarity scores in [0.0, 1.0].
        Note: FAISS with normalized embeddings produces L2 distance in [0, 4] where
        L2^2 = 2 * (1 - cos). Therefore cos = 1 - (L2^2 / 2).
        """
        if self.db is None:
            raise ValueError("FAISS index has not been built yet. Load or ingest papers first.")

        # FAISS search
        results_with_distance = self.db.similarity_search_with_score(query, k=k, filter=filter_dict)
        
        normalized_results: List[Tuple[Document, float]] = []
        for doc, distance in results_with_distance:
            # Convert L2 distance to cosine similarity
            # If distance is already L2 distance:
            cos_sim = max(0.0, min(1.0, 1.0 - (float(distance) ** 2) / 2.0)) if distance < 2.0 else max(0.0, 1.0 / (1.0 + float(distance)))
            normalized_results.append((doc, round(cos_sim, 4)))

        return normalized_results

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 15, lambda_mult: float = 0.5
    ) -> List[Document]:
        """Maximal Marginal Relevance search to balance relevance and diversity."""
        if self.db is None:
            raise ValueError("FAISS index is not initialized.")
        return self.db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

    def save(self, name: Optional[str] = None) -> str:
        """Persist FAISS index to disk."""
        if self.db is None:
            raise ValueError("No active FAISS index to save.")
        save_name = name or self.collection_name
        save_dir = os.path.join(self.index_dir, save_name)
        self.db.save_local(save_dir)
        logger.info(f"FAISS index persisted to {save_dir}")
        return save_dir

    def load(self, name: str) -> bool:
        """Load a persisted FAISS index from disk."""
        save_dir = os.path.join(self.index_dir, name)
        if not os.path.exists(save_dir):
            return False
        self.db = FAISS.load_local(save_dir, self.embeddings, allow_dangerous_deserialization=True)
        self.collection_name = name
        # Count documents inside docstore
        self.indexed_count = len(self.db.docstore._dict) if hasattr(self.db, "docstore") else 0
        logger.info(f"Loaded FAISS index '{name}' with {self.indexed_count} chunks.")
        return True
