"""Hybrid retrieval (BM25 + dense + RRF) unit tests with both lanes mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.tools._bm25 import BM25Okapi, _tokenize
from app.tools.vector_search import _rrf_merge


def test_bm25_ranks_relevant_doc_first():
    corpus = [
        "Apple Inc. announced quarterly revenue",
        "Bananas grow in tropical climates around the equator",
        "Apple Inc. revenue grew twelve percent year over year",
    ]
    bm25 = BM25Okapi([_tokenize(c) for c in corpus])
    top = bm25.top_k(_tokenize("apple revenue"), k=2)
    assert top[0][0] in (0, 2)
    assert top[1][0] in (0, 2)
    assert top[0][0] != top[1][0]


def test_rrf_prefers_docs_high_in_multiple_lanes():
    dense = [("a", {"content": "A"}), ("b", {"content": "B"}), ("c", {"content": "C"})]
    bm25 = [("c", {"content": "C"}), ("a", {"content": "A"}), ("d", {"content": "D"})]
    fused = _rrf_merge([dense, bm25])
    # 'a' appears at rank 1 of dense and rank 2 of bm25 → highest combined score.
    assert fused[0]["content"] == "A"
    # 'c' is the runner-up because it's near the top of both lanes.
    assert fused[1]["content"] == "C"


def test_vector_search_when_collection_missing():
    from app.tools.vector_search import vector_search

    with patch("app.tools.vector_search._get_client") as mock_client:
        client = MagicMock()
        client.collection_exists.return_value = False
        mock_client.return_value = client
        out = vector_search.invoke({"query": "anything"})
    assert "No documents indexed" in out


def test_vector_search_uses_hybrid_when_present():
    """Mock both dense and BM25 lanes and verify fusion + return shape."""
    import sys

    vs_mod = sys.modules["app.tools.vector_search"]

    fake_dense = [
        ("doc1", {"content": "Dense top: apple revenue ...", "metadata": {"source": "a.md", "date": "2024-01-01"}}),
        ("doc2", {"content": "Dense second: unrelated", "metadata": {"source": "b.md", "date": "2024-02-01"}}),
    ]
    fake_bm25 = [
        ("doc1", {"content": "Dense top: apple revenue ...", "metadata": {"source": "a.md", "date": "2024-01-01"}}),
        ("doc3", {"content": "BM25 second: keyword match", "metadata": {"source": "c.md", "date": "2024-03-01"}}),
    ]
    with (
        patch.object(vs_mod, "_get_client") as mock_client,
        patch.object(vs_mod, "_dense_lane", return_value=fake_dense),
        patch.object(vs_mod, "_bm25_lane", return_value=fake_bm25),
        patch.object(vs_mod, "_maybe_rerank", side_effect=lambda q, docs: docs),
    ):
        client = MagicMock()
        client.collection_exists.return_value = True
        mock_client.return_value = client
        out = vs_mod.vector_search.invoke({"query": "apple revenue"})
    assert "Doc 1" in out
    # doc1 should be top of fused → its source 'a.md' must appear before 'c.md'.
    assert out.index("a.md") < out.index("c.md")
