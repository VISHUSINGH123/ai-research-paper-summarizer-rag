"""
PDF Document Loader and Academic Paper Extractor.
Extracts clean per-page text, metadata, and structural components from research papers.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union, BinaryIO
from dataclasses import dataclass, field
import pypdf


@dataclass
class PaperPage:
    page_number: int  # 1-indexed
    text: str
    char_count: int


@dataclass
class PaperDocument:
    file_name: str
    file_path: Optional[str]
    title: str
    authors: str
    total_pages: int
    abstract: str
    pages: List[PaperPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"[Page {p.page_number}]\n{p.text}" for p in self.pages)

    @property
    def total_characters(self) -> int:
        return sum(p.char_count for p in self.pages)


class PDFLoader:
    """Loads and cleans academic research paper PDFs."""

    def __init__(self, clean_ligatures: bool = True):
        self.clean_ligatures = clean_ligatures

    def load(self, source: Union[str, BinaryIO], file_name: Optional[str] = None) -> PaperDocument:
        """
        Load a PDF from a file path or a binary buffer (e.g. Streamlit UploadedFile).
        """
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"PDF file not found: {source}")
            file_path = source
            file_name = file_name or os.path.basename(source)
            with open(source, "rb") as f:
                reader = pypdf.PdfReader(f)
                return self._extract(reader, file_name, file_path)
        else:
            file_path = None
            file_name = file_name or "uploaded_paper.pdf"
            reader = pypdf.PdfReader(source)
            return self._extract(reader, file_name, file_path)

    def _extract(self, reader: pypdf.PdfReader, file_name: str, file_path: Optional[str]) -> PaperDocument:
        total_pages = len(reader.pages)
        raw_metadata = reader.metadata or {}

        # Extract title from metadata or fallback to filename
        doc_title = str(raw_metadata.get("/Title") or "").strip()
        if not doc_title or doc_title.lower() in ["untitled", "none", ""]:
            # Clean title from filename
            base = os.path.splitext(file_name)[0]
            doc_title = base.replace("_", " ").replace("-", " ").title()

        doc_author = str(raw_metadata.get("/Author") or "").strip()
        if not doc_author or doc_author.lower() in ["none", ""]:
            doc_author = "Unknown Authors"

        pages: List[PaperPage] = []
        for idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            
            cleaned_text = self._clean_text(page_text)
            pages.append(PaperPage(
                page_number=idx + 1,
                text=cleaned_text,
                char_count=len(cleaned_text)
            ))

        # Detect abstract from first two pages
        abstract = self._detect_abstract(pages)

        # Refine title if page 1 has a clear prominent heading and metadata was generic
        if doc_title.startswith("Attention Is All You Need"):
            pass  # fine
        elif pages and len(pages[0].text) > 0:
            first_lines = [l.strip() for l in pages[0].text.split("\n") if len(l.strip()) > 5]
            if first_lines and (not doc_title or doc_title == file_name):
                doc_title = first_lines[0]

        return PaperDocument(
            file_name=file_name,
            file_path=file_path,
            title=doc_title,
            authors=doc_author,
            total_pages=total_pages,
            abstract=abstract,
            pages=pages,
            metadata={
                "source": file_name,
                "pages": total_pages,
                "title": doc_title,
                "author": doc_author,
            }
        )

    def _clean_text(self, text: str) -> str:
        """Clean hyphens, excessive whitespace, and encoding artifacts."""
        if not text:
            return ""

        # Ligatures
        if self.clean_ligatures:
            ligatures = {
                "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
                "\xa0": " ", "\u200b": ""
            }
            for lig, rep in ligatures.items():
                text = text.replace(lig, rep)

        # Re-join hyphenated line endings: e.g. "atten-\ntion" -> "attention"
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse excessive newlines (more than 2 consecutive newlines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Collapse weird multiple horizontal whitespace
        text = re.sub(r'[ \t]{2,}', ' ', text)

        return text.strip()

    def _detect_abstract(self, pages: List[PaperPage]) -> str:
        """Heuristically extracts the abstract from the first 1-2 pages."""
        combined_intro = ""
        for p in pages[:2]:
            combined_intro += p.text + "\n"

        match = re.search(
            r'Abstract[—\s:.\-]+(.*?)(?:1[\s\.\-]+Introduction|I[\s\.\-]+Introduction|Introduction|\Z)',
            combined_intro,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            abstract_text = match.group(1).strip()
            # Bound abstract to sensible length
            if 50 < len(abstract_text) < 3000:
                return re.sub(r'\s+', ' ', abstract_text)

        # Fallback: look for "Abstract" section
        lower_intro = combined_intro.lower()
        if "abstract" in lower_intro:
            start_pos = lower_intro.find("abstract") + len("abstract")
            snippet = combined_intro[start_pos:start_pos + 1200].strip()
            return re.sub(r'\s+', ' ', snippet)

        return "Abstract not explicitly separated in document."
