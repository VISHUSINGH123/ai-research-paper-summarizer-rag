"""
arXiv Fetcher and Search Utility.
Allows searching, fetching metadata, and downloading papers from arXiv by ID or keyword query.
"""

import os
import re
import urllib.request
import logging
from typing import List, Dict, Any, Optional
import arxiv

logger = logging.getLogger(__name__)


class ArxivFetcher:
    """Interacts with arXiv API to search and retrieve research papers."""

    def __init__(self, download_dir: str = "data/cache/arxiv_downloads"):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        self.client = arxiv.Client(
            page_size=10,
            delay_seconds=3.0,
            num_retries=3
        )

    @staticmethod
    def extract_arxiv_id(query: str) -> Optional[str]:
        """
        Extract arXiv ID from raw ID string or URL.
        Matches e.g. '1706.03762', '1706.03762v2', 'arxiv.org/abs/1706.03762'
        """
        query = query.strip()
        # Direct pattern e.g. 1706.03762 or 2106.09685
        direct_match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', query)
        if direct_match:
            return direct_match.group(1)

        # Old style arxiv id e.g. math/0211159, cs/0101010
        old_match = re.search(r'([a-zA-Z\-]+/\d{7})', query)
        if old_match:
            return old_match.group(1)

        return None

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers by keyword or title.
        """
        arxiv_id = self.extract_arxiv_id(query)
        if arxiv_id:
            return self.get_by_id(arxiv_id)

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        try:
            for r in self.client.results(search):
                results.append({
                    "id": r.get_short_id(),
                    "title": r.title.replace("\n", " ").strip(),
                    "authors": [a.name for a in r.authors],
                    "summary": r.summary.replace("\n", " ").strip(),
                    "published": r.published.strftime("%Y-%m-%d") if r.published else "",
                    "pdf_url": r.pdf_url,
                    "categories": r.categories,
                    "comment": r.comment or "",
                })
        except Exception as e:
            logger.error(f"Error querying arXiv API: {e}")

        return results

    def get_by_id(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """Fetch details for a specific arXiv ID."""
        clean_id = self.extract_arxiv_id(arxiv_id) or arxiv_id
        search = arxiv.Search(id_list=[clean_id])
        results = []
        try:
            for r in self.client.results(search):
                results.append({
                    "id": r.get_short_id(),
                    "title": r.title.replace("\n", " ").strip(),
                    "authors": [a.name for a in r.authors],
                    "summary": r.summary.replace("\n", " ").strip(),
                    "published": r.published.strftime("%Y-%m-%d") if r.published else "",
                    "pdf_url": r.pdf_url,
                    "categories": r.categories,
                    "comment": r.comment or "",
                })
        except Exception as e:
            logger.error(f"Error fetching paper {arxiv_id}: {e}")

        return results

    def download_pdf(self, arxiv_id_or_url: str) -> str:
        """
        Download arXiv PDF and save to download_dir. Returns local file path.
        """
        arxiv_id = self.extract_arxiv_id(arxiv_id_or_url)
        if not arxiv_id:
            raise ValueError(f"Could not parse valid arXiv ID from: {arxiv_id_or_url}")

        clean_id = arxiv_id.replace("/", "_")
        target_path = os.path.join(self.download_dir, f"{clean_id}.pdf")

        if os.path.exists(target_path) and os.path.getsize(target_path) > 10000:
            logger.info(f"Using cached arXiv PDF: {target_path}")
            return target_path

        download_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        logger.info(f"Downloading arXiv paper {arxiv_id} from {download_url}...")
        req = urllib.request.Request(download_url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            with open(target_path, "wb") as f:
                f.write(data)

        logger.info(f"Saved {target_path} ({len(data):,} bytes)")
        return target_path
