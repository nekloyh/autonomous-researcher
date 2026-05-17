"""Targeted extraction helpers for long documents and PDFs."""
from __future__ import annotations

import io
import re
from urllib.parse import urlparse

import requests

from app.tools.fetch_url import fetch_url

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]+")


def is_long_document_url(url: str) -> bool:
    path = urlparse(url or "").path.lower()
    return path.endswith(".pdf")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "") if len(t) > 2}


def _extract_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except Exception:
        return ""
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(f"\n\n## Page {i}\n{text}")
    return "\n".join(pages)


def _chunks(text: str, *, size: int = 3000, overlap: int = 300) -> list[tuple[str, str]]:
    cleaned = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    if not cleaned:
        return []
    out: list[tuple[str, str]] = []
    start = 0
    idx = 1
    while start < len(cleaned):
        chunk = cleaned[start : start + size]
        heading_match = re.search(r"(?m)^#{1,3}\s+(.+)$", chunk)
        label = heading_match.group(1).strip()[:80] if heading_match else f"chunk {idx}"
        out.append((label, chunk))
        idx += 1
        start += max(1, size - overlap)
    return out


def relevant_document_excerpt(url: str, target_terms: list[str], *, max_chars: int = 8000) -> str:
    """Return the most relevant long-document chunks for a research cell."""
    text = ""
    if is_long_document_url(url):
        try:
            resp = requests.get(
                url,
                timeout=20,
                headers={"User-Agent": "Mozilla/5.0 (ResearchAgent/1.0)"},
            )
            resp.raise_for_status()
            text = _extract_pdf_text(resp.content)
        except Exception as e:
            return f"ERROR: Could not extract PDF text from {url}. Reason: {e}"
    if not text:
        text = fetch_url.invoke({"url": url})
    query_tokens = _tokens(" ".join(target_terms))
    ranked = []
    for label, chunk in _chunks(text):
        chunk_tokens = _tokens(chunk)
        score = len(query_tokens & chunk_tokens)
        ranked.append((score, label, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[:3] if ranked else []
    if not selected:
        return text[:max_chars]
    parts = [
        f"### DOCUMENT SECTION: {label}\n{chunk[: max_chars // max(1, len(selected))]}"
        for score, label, chunk in selected
        if score > 0 or len(selected) == 1
    ]
    return "\n\n".join(parts)[:max_chars]
