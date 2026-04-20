"""PDF parsing utilities using pymupdf4llm.

Provides conversion of PDF documents to markdown text and extraction of
basic front-matter metadata (title, authors, abstract).
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFError(Exception):
    """Raised when PDF processing fails."""


class PDFTools:
    """Static methods for reading and extracting information from PDF files."""

    @staticmethod
    async def parse_pdf(pdf_path: Path) -> str:
        """Convert a PDF file to markdown text.

        Uses ``pymupdf4llm.to_markdown`` in a thread to avoid blocking.

        Raises :class:`PDFError` if conversion fails.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise PDFError(f"PDF file not found: {pdf_path}")

        def _convert() -> str:
            import pymupdf4llm

            return pymupdf4llm.to_markdown(str(pdf_path))

        try:
            return await asyncio.to_thread(_convert)
        except Exception as exc:
            raise PDFError(f"Failed to convert PDF to markdown: {exc}") from exc

    @staticmethod
    async def extract_metadata(pdf_path: Path) -> dict:
        """Extract basic metadata from a PDF's front matter.

        Attempts to read the PDF's embedded metadata and then parse the
        first page text for title, authors, abstract, and links.

        Returns a dict with keys: ``title``, ``authors``, ``abstract``, ``links``.
        Missing fields are returned as empty strings or empty lists.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise PDFError(f"PDF file not found: {pdf_path}")

        def _extract() -> dict:
            import pymupdf  # fitz

            doc = pymupdf.open(str(pdf_path))
            meta = doc.metadata or {}

            # Start with embedded metadata.
            title = meta.get("title", "").strip()
            authors = meta.get("author", "").strip()

            # Try to extract richer information from the first page text.
            first_page_text = ""
            if doc.page_count > 0:
                first_page_text = doc[0].get_text()

            # Extract links from all pages (limiting to first few pages).
            links: list[str] = []
            for page_idx in range(min(doc.page_count, 3)):
                page = doc[page_idx]
                for link in page.get_links():
                    uri = link.get("uri", "")
                    if uri and uri.startswith("http"):
                        links.append(uri)

            # If embedded title is missing, try to infer from first page.
            if not title and first_page_text:
                lines = [ln.strip() for ln in first_page_text.split("\n") if ln.strip()]
                if lines:
                    # Heuristic: the first substantial line is often the title.
                    title = lines[0]

            # Extract abstract from first page text.
            abstract = _extract_abstract(first_page_text)

            doc.close()

            return {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "links": links,
            }

        try:
            return await asyncio.to_thread(_extract)
        except Exception as exc:
            raise PDFError(f"Failed to extract PDF metadata: {exc}") from exc


def _extract_abstract(text: str) -> str:
    """Try to extract the abstract section from paper text."""
    if not text:
        return ""

    # Look for a section starting with "Abstract" and ending at the next section heading.
    # Common patterns: "Abstract", "ABSTRACT", "Abstract.", "Abstract—"
    match = re.search(
        r"(?:^|\n)\s*(?:abstract|ABSTRACT)[.:\s\u2014-]*\n?(.*?)(?=\n\s*(?:\d+\.?\s+)?(?:Introduction|INTRODUCTION|I\.\s|1\s)|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        abstract = match.group(1).strip()
        # Clean up excessive whitespace.
        abstract = re.sub(r"\s+", " ", abstract)
        return abstract

    return ""
