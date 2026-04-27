"""Long-term semantic memory: cache past (query, report) pairs in Qdrant.

If a new query is similar enough to a recent one, return the cached report.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.config import QDRANT_URL, get_embeddings

COLLECTION = "past_queries"
DEFAULT_THRESHOLD = 0.85
DEFAULT_TTL_DAYS = 7


class SemanticMemory:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD, ttl_days: int = DEFAULT_TTL_DAYS):
        self.threshold = threshold
        self.ttl_days = ttl_days
        self._client = QdrantClient(url=QDRANT_URL)
        self._emb = get_embeddings()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            if self._client.collection_exists(COLLECTION):
                return
            sample = self._emb.embed_query("dim probe")
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=len(sample), distance=Distance.COSINE),
            )
        except Exception:
            # Qdrant unreachable — memory is best-effort.
            pass

    def store(
        self,
        query: str,
        final_report: str,
        session_id: str,
        citations: list[str] | None = None,
    ) -> None:
        try:
            vec = self._emb.embed_query(query)
            self._client.upsert(
                collection_name=COLLECTION,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload={
                            "query": query,
                            "final_report": final_report,
                            "session_id": session_id,
                            "citations": citations or [],
                            "stored_at": datetime.now(UTC).isoformat(),
                        },
                    )
                ],
            )
        except Exception:
            pass

    def find_similar(self, query: str) -> dict[str, Any] | None:
        try:
            if not self._client.collection_exists(COLLECTION):
                return None
            vec = self._emb.embed_query(query)
            hits = self._client.search(
                collection_name=COLLECTION,
                query_vector=vec,
                limit=1,
                with_payload=True,
            )
        except Exception:
            return None
        if not hits:
            return None
        hit = hits[0]
        if hit.score < self.threshold:
            return None

        payload = hit.payload or {}
        stored_at = payload.get("stored_at")
        age_days = 0
        if stored_at:
            try:
                ts = datetime.fromisoformat(stored_at)
                age = datetime.now(UTC) - ts
                if age > timedelta(days=self.ttl_days):
                    return None
                age_days = age.days
            except ValueError:
                return None
        return {
            "score": float(hit.score),
            "age_days": age_days,
            "final_report": payload.get("final_report", ""),
            "citations": payload.get("citations", []),
            "session_id": payload.get("session_id", ""),
        }
