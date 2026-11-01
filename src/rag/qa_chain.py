"""
Grounded Question Answering Pipeline for Research Papers.
Retrieves semantic chunks, structures citation references, and produces grounded responses.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from langchain_core.documents import Document

from ..vectorstores.store_manager import VectorStoreManager, StoreType
from ..llm.llm_provider import LLMProvider
from ..llm.prompts import GROUNDED_QA_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    page: int
    section: str
    similarity: float
    source: str
    preview: str
    chunk_index: int


@dataclass
class QAResult:
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    engine_used: str = "FAISS"
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    top_similarity: float = 0.0
    chunks_retrieved: int = 0


class ResearchPaperQA:
    """Grounded RAG Question-Answering engine."""

    def __init__(self, store_manager: VectorStoreManager, llm_provider: LLMProvider):
        self.store_manager = store_manager
        self.llm_provider = llm_provider

    def ask(
        self,
        question: str,
        k: int = 5,
        engine: Optional[StoreType] = None,
        min_similarity: float = 0.25
    ) -> QAResult:
        """
        Ask a paper-specific question and receive a grounded answer with citations.
        """
        active_engine = engine or self.store_manager.active_engine

        # 1. Retrieval
        t_retrieval_start = time.perf_counter()
        raw_results = self.store_manager.similarity_search_with_score(
            query=question, k=k, engine=active_engine
        )
        retrieval_latency_ms = (time.perf_counter() - t_retrieval_start) * 1000

        # Filter by similarity if needed
        filtered_results = [(doc, sim) for doc, sim in raw_results if sim >= min_similarity]
        if not filtered_results and raw_results:
            # Keep top-2 if all are below threshold to avoid complete emptiness
            filtered_results = raw_results[:2]

        # 2. Format Context & Citations
        citations: List[Citation] = []
        context_blocks = []

        for doc, sim in filtered_results:
            page = doc.metadata.get("page", 1)
            section = doc.metadata.get("section", "General")
            source = doc.metadata.get("source", "paper.pdf")
            c_idx = doc.metadata.get("chunk_index", 0)

            citations.append(Citation(
                page=page,
                section=section,
                similarity=sim,
                source=source,
                preview=doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
                chunk_index=c_idx
            ))

            header = f"[Page {page} - {section} (Similarity: {sim * 100:.1f}%)]"
            context_blocks.append(f"{header}\n{doc.page_content}")

        context_str = "\n\n---\n\n".join(context_blocks)

        # 3. LLM Generation
        prompt = GROUNDED_QA_PROMPT.format(context=context_str, question=question)

        t_gen_start = time.perf_counter()
        answer = self.llm_provider.generate(prompt=prompt)
        gen_latency_ms = (time.perf_counter() - t_gen_start) * 1000

        top_score = filtered_results[0][1] if filtered_results else 0.0

        return QAResult(
            question=question,
            answer=answer,
            citations=citations,
            engine_used=active_engine,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(gen_latency_ms, 2),
            total_latency_ms=round(retrieval_latency_ms + gen_latency_ms, 2),
            top_similarity=round(top_score, 4),
            chunks_retrieved=len(filtered_results)
        )
