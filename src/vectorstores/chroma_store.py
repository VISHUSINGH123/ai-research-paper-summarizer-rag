"""
ChromaDB Vector Store wrapper for persistent document collections.
"""

import os
import time
import shutil
import logging
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class ChromaStore:
    """ChromaDB collection manager with persistent disk backend."""

    def __init__(self, embeddings: Embeddings, persist_dir: str = "data/cache/chroma_db"):
        self.embeddings = embeddings
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.db: Optional[Chroma] = None
        self.collection_name: str = "default_papers"
        self.indexed_count: int = 0

    def build_index(self, documents: List[Document], collection_name: str = "default_papers") -> Dict[str, Any]:
        """
        Build or replace a persistent Chroma collection from documents.
        """
        start_time = time.perf_counter()
        logger.info(f"Building ChromaDB collection '{collection_name}' with {len(documents)} chunks...")

        # Sanitize collection name for Chroma: 3-63 chars, alphanumeric, dots, underscores, hyphens
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in collection_name)[:63]
        if len(safe_name) < 3:
            safe_name = f"col_{safe_name}"

        self.collection_name = safe_name
        collection_persist_dir = os.path.join(self.persist_dir, safe_name)
        os.makedirs(collection_persist_dir, exist_ok=True)

        self.db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=safe_name,
            persist_directory=collection_persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )
        self.indexed_count = len(documents)
        elapsed_sec = time.perf_counter() - start_time

        logger.info(f"Chroma collection built in {elapsed_sec:.3f}s for {len(documents)} items.")
        return {
            "engine": "ChromaDB",
            "collection_name": safe_name,
            "count": len(documents),
            "indexing_time_sec": elapsed_sec,
            "persist_directory": collection_persist_dir
        }

    def load_existing(self, collection_name: str = "default_papers") -> bool:
        """Connect to an existing persisted collection."""
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in collection_name)[:63]
        collection_persist_dir = os.path.join(self.persist_dir, safe_name)
        if not os.path.exists(collection_persist_dir):
            return False

        try:
            self.collection_name = safe_name
            self.db = Chroma(
                collection_name=safe_name,
                embedding_function=self.embeddings,
                persist_directory=collection_persist_dir,
                collection_metadata={"hnsw:space": "cosine"}
            )
            # Count elements
            count = self.db._collection.count()
            self.indexed_count = count
            logger.info(f"Loaded existing Chroma collection '{safe_name}' with {count} items.")
            return True
        except Exception as e:
            logger.error(f"Error loading Chroma collection '{safe_name}': {e}")
            return False

    def add_documents(self, documents: List[Document]) -> int:
        """Add additional documents to existing Chroma collection."""
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
        Search Chroma collection.
        When distance metric is cosine, Chroma returns cosine distance (1 - cos).
        We convert distance to cosine similarity: cos = 1.0 - distance.
        """
        if self.db is None:
            raise ValueError("Chroma collection has not been initialized.")

        results_with_distance = self.db.similarity_search_with_score(
            query=query, k=k, filter=filter_dict
        )

        normalized_results: List[Tuple[Document, float]] = []
        for doc, distance in results_with_distance:
            # Cosine distance to similarity:
            cos_sim = max(0.0, min(1.0, 1.0 - float(distance)))
            normalized_results.append((doc, round(cos_sim, 4)))

        return normalized_results

    def reset_collection(self, collection_name: Optional[str] = None):
        """Remove persisted data for collection."""
        name = collection_name or self.collection_name
        col_dir = os.path.join(self.persist_dir, name)
        if os.path.exists(col_dir):
            try:
                shutil.rmtree(col_dir)
            except Exception as e:
                logger.warning(f"Could not remove Chroma dir {col_dir}: {e}")
        self.db = None
        self.indexed_count = 0
