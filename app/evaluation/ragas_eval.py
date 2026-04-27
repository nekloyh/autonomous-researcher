"""RAGAS evaluation pipeline.

Builds a Hugging Face Dataset from agent runs, then scores faithfulness,
answer relevancy, context precision, and context recall using Gemini as judge.
"""
from __future__ import annotations

from typing import Any


def build_record(query: str, final_report: str, findings: list[dict], ground_truth: str | None = None) -> dict[str, Any]:
    contexts = [(f.get("content") or "") for f in findings if f.get("content")]
    return {
        "question": query,
        "answer": final_report,
        "contexts": contexts or [final_report[:1000]],
        "ground_truth": ground_truth or "",
    }


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run RAGAS on a list of records. Returns aggregate scores.

    Imports lazily so the module loads even when the [eval] extras aren't installed.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        return {"error": f"RAGAS extras not installed: {e}. Run 'uv sync --extra eval'."}

    from app.config import get_critic_llm, get_embeddings

    ds = Dataset.from_list(records)
    metrics = [faithfulness, answer_relevancy, context_precision]
    if any(r.get("ground_truth") for r in records):
        metrics.append(context_recall)

    result = evaluate(
        ds,
        metrics=metrics,
        llm=get_critic_llm(),
        embeddings=get_embeddings(),
    )
    try:
        df = result.to_pandas()
        per_query = df.to_dict(orient="records")
    except Exception:
        per_query = []
    aggregate = {k: float(v) for k, v in result._repr_dict.items()} if hasattr(result, "_repr_dict") else {}
    return {"aggregate": aggregate, "per_query": per_query}
