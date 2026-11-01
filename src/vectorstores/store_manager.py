"""
Unified Vector Store Manager.
Coordinates FAISS and ChromaDB stores, provides dual-indexing, engine toggling,
and side-by-side performance benchmarking.
"""

import time
import logging
from typing import List, Tuple, Dict, Any, Optional, Literal
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .faiss_store import FAISSStore
from .chroma_store import ChromaStore

logger = logging.getLogger(__name__)

StoreType = Literal["FAISS", "ChromaDB"]


class VectorStoreManager:
    """Manages both FAISS and ChromaDB backends with a unified interface."""

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.faiss_store = FAISSStore(embeddings=embeddings)
        self.chroma_store = ChromaStore(embeddings=embeddings)
        self.active_engine: StoreType = "FAISS"
        self.current_paper_name: Optional[str] = None
        self.indexed_documents: List[Document] = []

    def set_active_engine(self, engine: StoreType):
        if engine not in ("FAISS", "ChromaDB"):
            raise ValueError(f"Unsupported vector engine: {engine}. Choose 'FAISS' or 'ChromaDB'.")
        self.active_engine = engine
        logger.info(f"Active vector store set to: {self.active_engine}")

    def index_documents(
        self,
        documents: List[Document],
        collection_name: str = "paper_collection",
        target: Literal["both", "FAISS", "ChromaDB"] = "both"
    ) -> Dict[str, Any]:
        """
        Index documents into FAISS, ChromaDB, or both.
        """
        self.indexed_documents = documents
        self.current_paper_name = collection_name
        summary = {"documents_count": len(documents), "engines": {}}

        if target in ("both", "FAISS"):
            faiss_res = self.faiss_store.build_index(documents, collection_name=collection_name)
            summary["engines"]["FAISS"] = faiss_res

        if target in ("both", "ChromaDB"):
            chroma_res = self.chroma_store.build_index(documents, collection_name=collection_name)
            summary["engines"]["ChromaDB"] = chroma_res

        return summary

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        engine: Optional[StoreType] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search using the specified or active engine.
        Returns list of (Document, similarity_score) tuples where similarity_score is in [0, 1].
        """
        target_engine = engine or self.active_engine
        if target_engine == "FAISS":
            return self.faiss_store.similarity_search_with_score(query, k=k)
        elif target_engine == "ChromaDB":
            return self.chroma_store.similarity_search_with_score(query, k=k)
        else:
            raise ValueError(f"Unknown engine: {target_engine}")

    def benchmark(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Run side-by-side benchmarking between FAISS and ChromaDB on the same query.
        Measures latency, top similarity scores, and chunk overlap.
        """
        # Benchmark FAISS
        t0 = time.perf_counter()
        faiss_results = self.faiss_store.similarity_search_with_score(query, k=k)
        faiss_latency_ms = (time.perf_counter() - t0) * 1000

        # Benchmark ChromaDB
        t1 = time.perf_counter()
        chroma_results = self.chroma_store.similarity_search_with_score(query, k=k)
        chroma_latency_ms = (time.perf_counter() - t1) * 1000

        # Compare overlap
        faiss_contents = {doc.page_content.strip() for doc, _ in faiss_results}
        chroma_contents = {doc.page_content.strip() for doc, _ in chroma_results}
        common_chunks = faiss_contents.intersection(chroma_contents)
        overlap_percentage = (len(common_chunks) / k) * 100 if k > 0 else 0

        faiss_avg_score = (
            sum(score for _, score in faiss_results) / len(faiss_results)
            if faiss_results else 0.0
        )
        chroma_avg_score = (
            sum(score for _, score in chroma_results) / len(chroma_results)
            if chroma_results else 0.0
        )

        return {
            "query": query,
            "k": k,
            "faiss": {
                "latency_ms": round(faiss_latency_ms, 2),
                "top_score": faiss_results[0][1] if faiss_results else 0.0,
                "avg_score": round(faiss_avg_score, 4),
                "count": len(faiss_results),
            },
            "chroma": {
                "latency_ms": round(chroma_latency_ms, 2),
                "top_score": chroma_results[0][1] if chroma_results else 0.0,
                "avg_score": round(chroma_avg_score, 4),
                "count": len(chroma_results),
            },
            "overlap_count": len(common_chunks),
            "overlap_percentage": round(overlap_percentage, 1),
            "faster_engine": "FAISS" if faiss_latency_ms < chroma_latency_ms else "ChromaDB",
            "speedup_ratio": round(
                max(faiss_latency_ms, chroma_latency_ms) / (min(faiss_latency_ms, chroma_latency_ms) or 0.001), 2
            ),
        }
