"""Semantic search over local document corpus (Qdrant).

Pipeline: dense (Qdrant ANN) + BM25 (in-memory) → Reciprocal Rank Fusion → optional
local cross-encoder rerank → top-3.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import QDRANT_URL, get_embeddings

COLLECTION = "research_corpus"

DENSE_K = 10
BM25_K = 10
FUSED_K = 10
RRF_K_CONST = 60
RETURN_K = 3


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=_get_client(),
        collection_name=COLLECTION,
        embedding=get_embeddings(),
    )


def _doc_key(meta: dict, content: str) -> str:
    """Stable identity for a chunk across lanes. Prefer source+offset; fall
    back to first 80 chars of content."""
    src = meta.get("source") or ""
    page = meta.get("page") or meta.get("offset") or ""
    if src:
        return f"{src}::{page}::{content[:60]}"
    return content[:80]


def _rrf_merge(rankings: list[list[tuple[str, dict]]], k_const: int = RRF_K_CONST) -> list[dict]:
    """Reciprocal rank fusion across lanes. Each lane is a ranked list of
    (doc_key, doc_dict). Returns merged docs sorted by RRF score."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for lane in rankings:
        for rank, (key, doc) in enumerate(lane):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_const + rank + 1)
            payloads.setdefault(key, doc)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [payloads[k] for k, _ in ordered]


def _dense_lane(query: str) -> list[tuple[str, dict]]:
    try:
        vs = get_vector_store()
        docs = vs.similarity_search(query, k=DENSE_K)
    except Exception:
        return []
    lane: list[tuple[str, dict]] = []
    for d in docs:
        meta = d.metadata or {}
        content = d.page_content or ""
        key = _doc_key(meta, content)
        lane.append((key, {"content": content, "metadata": meta}))
    return lane


def _bm25_lane(query: str) -> list[tuple[str, dict]]:
    try:
        from app.tools._bm25 import bm25_search
    except Exception:
        return []
    try:
        records = bm25_search(query, k=BM25_K)
    except Exception:
        return []
    lane: list[tuple[str, dict]] = []
    for r in records:
        meta = r.get("metadata") or {}
        content = r.get("content") or ""
        lane.append((_doc_key(meta, content), {"content": content, "metadata": meta}))
    return lane


def _maybe_rerank(query: str, docs: list[dict]) -> list[dict]:
    """Local cross-encoder rerank. Best-effort: returns input order if the
    model isn't available (first-run download, no network, etc.)."""
    if len(docs) <= RETURN_K:
        return docs
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        encoder = _get_reranker(TextCrossEncoder)
        passages = [d["content"] for d in docs]
        scored = list(encoder.rerank(query=query, documents=passages))
        scored_pairs = sorted(zip(scored, docs, strict=True), key=lambda p: p[0], reverse=True)
        return [d for _, d in scored_pairs]
    except Exception:
        return docs


@lru_cache(maxsize=1)
def _get_reranker(cls):
    # fastembed downloads on first instantiation; small model keeps it free.
    return cls(model_name="Xenova/ms-marco-MiniLM-L-6-v2")


@tool
def vector_search(query: str) -> str:
    """Search internal document corpus using hybrid retrieval (dense + BM25 + rerank).

    Use this for:
    - Domain-specific documents already ingested into the corpus
    - Historical data / archived reports
    - When web search isn't appropriate or returns nothing

    Args:
        query: Natural language search query

    Returns:
        Top 3 most relevant document excerpts with source/date metadata,
        or a friendly message when the collection is empty / unavailable.
    """
    try:
        client = _get_client()
        if not client.collection_exists(COLLECTION):
            return (
                f"No documents indexed yet. The '{COLLECTION}' collection does "
                "not exist. Use `web_search` instead."
            )
    except Exception as e:
        return f"ERROR: Vector store unavailable: {type(e).__name__}: {e}"

    lanes = [lane for lane in (_dense_lane(query), _bm25_lane(query)) if lane]
    if not lanes:
        return f"No documents found matching: {query}"

    fused = _rrf_merge(lanes)[:FUSED_K]
    reranked = _maybe_rerank(query, fused)[:RETURN_K]

    formatted: list[str] = []
    for i, d in enumerate(reranked, 1):
        meta = d.get("metadata") or {}
        content = d.get("content") or ""
        snippet = content[:500]
        if len(content) > 500:
            snippet += "…"
        formatted.append(
            f"[Doc {i}] source={meta.get('source', 'unknown')} "
            f"date={meta.get('date', 'n/a')}\n"
            f"---\n{snippet}"
        )
    return "\n\n".join(formatted) if formatted else f"No documents found matching: {query}"
