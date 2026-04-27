"""Ingest local .md/.txt corpus into Qdrant.

Usage:
    uv run python scripts/ingest_corpus.py [--corpus-dir corpus/] [--reset]
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.config import QDRANT_URL, get_embeddings
from app.tools.vector_search import COLLECTION, get_vector_store


def load_docs(folder: Path) -> list[Document]:
    docs: list[Document] = []
    for path in folder.rglob("*"):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(path.relative_to(folder)),
                    "date": mtime.date().isoformat(),
                },
            )
        )
    return docs


def ensure_collection(reset: bool = False) -> None:
    client = QdrantClient(url=QDRANT_URL)
    exists = client.collection_exists(COLLECTION)
    if exists and reset:
        client.delete_collection(COLLECTION)
        exists = False
    if not exists:
        emb = get_embeddings()
        sample = emb.embed_query("dimension probe")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=len(sample), distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION}' (dim={len(sample)})")


def main(corpus_dir: str, reset: bool) -> None:
    folder = Path(corpus_dir)
    if not folder.exists():
        raise SystemExit(f"Corpus folder not found: {folder}")

    ensure_collection(reset=reset)

    docs = load_docs(folder)
    print(f"Loaded {len(docs)} documents from {folder}")
    if not docs:
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    vs = get_vector_store()
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vs.add_documents(batch)
        print(f"  Ingested {i + len(batch)}/{len(chunks)}")
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", default="./corpus")
    p.add_argument("--reset", action="store_true", help="Delete and recreate collection")
    args = p.parse_args()
    main(args.corpus_dir, args.reset)
