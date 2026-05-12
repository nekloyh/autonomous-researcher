"""Tiny BM25Okapi over a list of pre-tokenised documents.

We avoid a hard dep on `rank-bm25`: it's listed in pyproject for parity with the
plan, but this implementation is pure-Python so the corpus retrieval path keeps
working even if the dep isn't installed yet. ~30 lines, no surprises.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from qdrant_client.http.models import Filter

from app.config import QDRANT_URL
from app.tools.vector_search import COLLECTION

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Okapi:
    """Lightweight BM25 Okapi scorer.

    Reference: https://en.wikipedia.org/wiki/Okapi_BM25
    """

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter] = [Counter(toks) for toks in corpus_tokens]
        self.doc_len: list[int] = [len(toks) for toks in corpus_tokens]
        n_docs = len(corpus_tokens) or 1
        self.avgdl = sum(self.doc_len) / n_docs
        df: Counter = Counter()
        for toks in corpus_tokens:
            for term in set(toks):
                df[term] += 1
        self.idf: dict[str, float] = {
            term: math.log((n_docs - cnt + 0.5) / (cnt + 0.5) + 1.0) for term, cnt in df.items()
        }

    def score(self, query_tokens: list[str], idx: int) -> float:
        f = self.doc_freqs[idx]
        dl = self.doc_len[idx] or 1
        s = 0.0
        for q in query_tokens:
            if q not in f:
                continue
            idf = self.idf.get(q, 0.0)
            tf = f[q]
            s += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        scored = [(i, self.score(query_tokens, i)) for i in range(len(self.doc_freqs))]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:k]


@lru_cache(maxsize=1)
def _load_corpus() -> tuple[BM25Okapi, list[dict]] | None:
    """Pull every chunk from Qdrant once, build BM25 in memory.

    Returns None if Qdrant is unreachable or the collection is empty — callers
    should treat that as "BM25 lane unavailable" and fall back to dense-only."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_URL)
        if not client.collection_exists(COLLECTION):
            return None
        records: list[dict] = []
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=COLLECTION,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=offset,
                scroll_filter=Filter(),
            )
            for point in batch:
                payload = point.payload or {}
                content = payload.get("page_content") or payload.get("content") or ""
                if not content:
                    continue
                records.append({"id": point.id, "content": content, "metadata": payload})
            if offset is None:
                break
        if not records:
            return None
        tokens = [_tokenize(r["content"]) for r in records]
        return BM25Okapi(tokens), records
    except Exception:
        return None


def reset_cache() -> None:
    _load_corpus.cache_clear()


def bm25_search(query: str, k: int = 10) -> list[dict]:
    """Return up to k payload dicts ranked by BM25, or [] when unavailable."""
    state = _load_corpus()
    if not state:
        return []
    bm25, records = state
    ranking = bm25.top_k(_tokenize(query), k=k)
    return [records[i] for i, _ in ranking if i < len(records)]
