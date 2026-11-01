"""
Multi-Section Research Paper Summarizer.
Uses targeted aspect-based retrieval to construct grounded, high-fidelity academic summaries.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from langchain_core.documents import Document

from ..vectorstores.store_manager import VectorStoreManager, StoreType
from ..llm.llm_provider import LLMProvider
from ..llm.prompts import EXECUTIVE_SUMMARY_PROMPT, MULTI_SECTION_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    paper_title: str
    mode: str  # "executive" or "multi_section"
    summary_text: str
    engine_used: str
    aspect_chunks_used: int
    latency_ms: float
    sections: Dict[str, str] = field(default_factory=dict)


class PaperSummarizer:
    """Summarizes research papers using grounded multi-hop RAG."""

    # Targeted query facets to extract key information across the whole paper
    ASPECT_QUERIES = {
        "motivation": "What is the primary problem statement, motivation, and limitation of prior work?",
        "contribution": "What are the core novel contributions, key hypotheses, and principal breakthroughs?",
        "methodology": "What is the model architecture, mathematical formulation, equations, and algorithm design?",
        "results": "What are the experimental results, benchmark comparisons, datasets, and performance metrics?",
        "limitations": "What are the limitations, computational constraints, failure modes, and future work?",
    }

    def __init__(self, store_manager: VectorStoreManager, llm_provider: LLMProvider):
        self.store_manager = store_manager
        self.llm_provider = llm_provider

    def summarize(
        self,
        paper_title: Optional[str] = None,
        mode: str = "multi_section",  # "executive" or "multi_section"
        engine: Optional[StoreType] = None
    ) -> SummaryResult:
        """
        Generate a comprehensive research paper summary.
        """
        active_engine = engine or self.store_manager.active_engine
        t_start = time.perf_counter()

        # Gather aspect-targeted context chunks
        collected_docs: List[Document] = []
        seen_contents = set()

        # If executive summary, query overview aspects
        queries_to_run = (
            ["motivation", "contribution", "results"]
            if mode == "executive"
            else list(self.ASPECT_QUERIES.keys())
        )

        for aspect in queries_to_run:
            query_str = self.ASPECT_QUERIES[aspect]
            results = self.store_manager.similarity_search_with_score(
                query=query_str, k=2, engine=active_engine
            )
            for doc, _ in results:
                content = doc.page_content.strip()
                if content not in seen_contents:
                    seen_contents.add(content)
                    collected_docs.append(doc)

        # Sort chunks by page number for natural reading flow
        collected_docs.sort(key=lambda d: d.metadata.get("page", 0))

        # Format context
        context_blocks = []
        for doc in collected_docs:
            page = doc.metadata.get("page", 1)
            sec = doc.metadata.get("section", "General")
            context_blocks.append(f"[Page {page} - {sec}]\n{doc.page_content}")

        context_str = "\n\n---\n\n".join(context_blocks)

        # Choose prompt
        if mode == "executive":
            prompt = EXECUTIVE_SUMMARY_PROMPT.format(context=context_str)
        else:
            prompt = MULTI_SECTION_SUMMARY_PROMPT.format(context=context_str)

        # Generate summary
        summary_text = self.llm_provider.generate(prompt=prompt, temperature=0.2)
        total_latency_ms = (time.perf_counter() - t_start) * 1000

        title = paper_title or self.store_manager.current_paper_name or "Research Paper"

        return SummaryResult(
            paper_title=title,
            mode=mode,
            summary_text=summary_text,
            engine_used=active_engine,
            aspect_chunks_used=len(collected_docs),
            latency_ms=round(total_latency_ms, 2)
        )
