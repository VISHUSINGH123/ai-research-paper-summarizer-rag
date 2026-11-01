"""
Vector Store Benchmark & Evaluation Suite.
Compares FAISS and ChromaDB retrieval latency, similarity distribution, and chunk overlap.
"""

import time
import logging
import pandas as pd
from typing import List, Dict, Any, Optional

from ..vectorstores.store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

STANDARD_BENCHMARK_QUERIES = [
    "What is the mathematical formulation of the attention mechanism?",
    "How does low-rank matrix decomposition reduce trainable parameters?",
    "What benchmarks and datasets were evaluated in the experiments?",
    "What are the computational complexity and memory requirements?",
    "What are the main limitations and open questions described by the authors?"
]


class VectorStoreBenchmark:
    """Runs repeatable benchmarks between FAISS and ChromaDB backends."""

    def __init__(self, store_manager: VectorStoreManager):
        self.store_manager = store_manager

    def run_benchmark(
        self, queries: Optional[List[str]] = None, k: int = 5, iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Execute comprehensive retrieval benchmark across FAISS and ChromaDB.
        """
        test_queries = queries or STANDARD_BENCHMARK_QUERIES
        results_rows = []

        faiss_latencies = []
        chroma_latencies = []
        overlap_scores = []

        for q in test_queries:
            # Measure FAISS over iterations
            f_times = []
            f_top_score = 0.0
            f_docs = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                res = self.store_manager.similarity_search_with_score(q, k=k, engine="FAISS")
                f_times.append((time.perf_counter() - t0) * 1000)
                if res:
                    f_top_score = res[0][1]
                    f_docs = [d.page_content.strip() for d, _ in res]

            avg_faiss_ms = sum(f_times) / len(f_times)
            faiss_latencies.append(avg_faiss_ms)

            # Measure ChromaDB over iterations
            c_times = []
            c_top_score = 0.0
            c_docs = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                res = self.store_manager.similarity_search_with_score(q, k=k, engine="ChromaDB")
                c_times.append((time.perf_counter() - t0) * 1000)
                if res:
                    c_top_score = res[0][1]
                    c_docs = [d.page_content.strip() for d, _ in res]

            avg_chroma_ms = sum(c_times) / len(c_times)
            chroma_latencies.append(avg_chroma_ms)

            # Calculate overlap
            common = len(set(f_docs).intersection(set(c_docs)))
            overlap_pct = (common / k * 100) if k > 0 else 0
            overlap_scores.append(overlap_pct)

            speedup = (
                f"{avg_chroma_ms / avg_faiss_ms:.2f}x (FAISS)"
                if avg_faiss_ms < avg_chroma_ms
                else f"{avg_faiss_ms / avg_chroma_ms:.2f}x (Chroma)"
            )

            results_rows.append({
                "Query": q[:45] + "..." if len(q) > 45 else q,
                "FAISS Latency (ms)": round(avg_faiss_ms, 2),
                "Chroma Latency (ms)": round(avg_chroma_ms, 2),
                "FAISS Top Sim": round(f_top_score, 4),
                "Chroma Top Sim": round(c_top_score, 4),
                "Overlap (%)": f"{overlap_pct:.0f}%",
                "Winner": "FAISS" if avg_faiss_ms < avg_chroma_ms else "ChromaDB"
            })

        df = pd.DataFrame(results_rows)

        avg_faiss_overall = sum(faiss_latencies) / len(faiss_latencies)
        avg_chroma_overall = sum(chroma_latencies) / len(chroma_latencies)
        avg_overlap = sum(overlap_scores) / len(overlap_scores)

        return {
            "summary_table": df,
            "overall": {
                "faiss_avg_latency_ms": round(avg_faiss_overall, 2),
                "chroma_avg_latency_ms": round(avg_chroma_overall, 2),
                "speedup_factor": round(
                    max(avg_faiss_overall, avg_chroma_overall) / (min(avg_faiss_overall, avg_chroma_overall) or 0.001), 2
                ),
                "faster_overall": "FAISS" if avg_faiss_overall < avg_chroma_overall else "ChromaDB",
                "average_overlap_pct": round(avg_overlap, 1),
                "queries_evaluated": len(test_queries),
                "total_documents_indexed": len(self.store_manager.indexed_documents),
            }
        }
