"""Semantic search over local document corpus (Qdrant)."""
from functools import lru_cache

from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import QDRANT_URL, get_embeddings

COLLECTION = "research_corpus"


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


@tool
def vector_search(query: str) -> str:
    """Search internal document corpus using semantic similarity.

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
        vs = get_vector_store()
        docs = vs.similarity_search(query, k=3)
    except Exception as e:
        return f"ERROR: Vector store unavailable: {type(e).__name__}: {e}"

    if not docs:
        return f"No documents found matching: {query}"

    formatted = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        snippet = d.page_content[:500]
        if len(d.page_content) > 500:
            snippet += "…"
        formatted.append(
            f"[Doc {i}] source={meta.get('source', 'unknown')} "
            f"date={meta.get('date', 'n/a')}\n"
            f"---\n{snippet}"
        )
    return "\n\n".join(formatted)
