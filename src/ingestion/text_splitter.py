"""
Academic Paper Text Splitter.
Splits research paper content into semantically rich chunks with section detection
and page-level citation metadata.
"""

import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .pdf_loader import PaperDocument, PaperPage


class AcademicTextSplitter:
    """
    Intelligent chunking for scientific and AI research papers.
    Preserves section context, page numbers, and paper metadata.
    """

    # Common scientific paper section headings
    SECTION_PATTERNS = [
        (r'(?i)\b(abstract)\b', "Abstract"),
        (r'(?i)\b(\d+[\.\s]+)?(introduction)\b', "Introduction"),
        (r'(?i)\b(\d+[\.\s]+)?(related\s+work|background|prior\s+work)\b', "Background & Related Work"),
        (r'(?i)\b(\d+[\.\s]+)?(method|methodology|model|architecture|approach|formulation|proposed\s+method)\b', "Methodology"),
        (r'(?i)\b(\d+[\.\s]+)?(experiment|experimental\s+setup|evaluation|implementation)\b', "Experiments"),
        (r'(?i)\b(\d+[\.\s]+)?(result|results|analysis|empirical\s+evaluation|ablation)\b', "Results & Analysis"),
        (r'(?i)\b(\d+[\.\s]+)?(discussion|limitations?)\b', "Discussion & Limitations"),
        (r'(?i)\b(\d+[\.\s]+)?(conclusion|concluding\s+remarks|future\s+work)\b', "Conclusion"),
        (r'(?i)\b(references|bibliography)\b', "References"),
    ]

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        keep_references: bool = False
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_references = keep_references
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
            length_function=len
        )

    def _detect_section(self, text: str, current_section: str) -> str:
        """Scan first lines of text chunk for section headings."""
        first_few_lines = "\n".join(text.split("\n")[:3])
        for pattern, section_name in self.SECTION_PATTERNS:
            if re.search(pattern, first_few_lines):
                return section_name
        return current_section

    def split_paper(self, paper: PaperDocument) -> List[Document]:
        """
        Split a PaperDocument into LangChain Document objects with rich metadata.
        """
        documents: List[Document] = []
        current_section = "Abstract"
        global_chunk_idx = 0

        for page in paper.pages:
            # Check if this page starts the references section
            if not self.keep_references and ("references" in page.text.lower() or "bibliography" in page.text.lower()):
                # If page is predominantly references, we can tag it or skip excessive reference pages
                if re.search(r'(?i)^\s*(references|bibliography)\s*$', page.text, re.MULTILINE):
                    current_section = "References"

            if not self.keep_references and current_section == "References":
                # Still include first reference page with brief citation info, but avoid noise
                pass

            page_chunks = self.splitter.split_text(page.text)
            for c_idx, chunk_text in enumerate(page_chunks):
                chunk_text = chunk_text.strip()
                if len(chunk_text) < 40:
                    continue  # skip tiny fragments / headers / page numbers

                current_section = self._detect_section(chunk_text, current_section)

                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": paper.file_name,
                        "paper_title": paper.title,
                        "authors": paper.authors,
                        "page": page.page_number,
                        "total_pages": paper.total_pages,
                        "chunk_index": global_chunk_idx,
                        "page_chunk_index": c_idx,
                        "section": current_section,
                        "char_count": len(chunk_text),
                        "token_estimate": max(1, len(chunk_text) // 4)
                    }
                )
                documents.append(doc)
                global_chunk_idx += 1

        return documents

    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Fallback splitter for raw text strings."""
        raw_chunks = self.splitter.split_text(text)
        base_meta = metadata or {}
        docs = []
        for idx, chunk in enumerate(raw_chunks):
            meta = {**base_meta, "chunk_index": idx, "char_count": len(chunk)}
            docs.append(Document(page_content=chunk, metadata=meta))
        return docs
