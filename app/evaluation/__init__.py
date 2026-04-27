"""Evaluation harness: heuristics + RAGAS metrics + benchmark test set."""
from app.evaluation.heuristics import run_heuristic_checks
from app.evaluation.ragas_eval import build_record, evaluate_records
from app.evaluation.test_set import TEST_SET, TestQuery, by_id, categories

__all__ = [
    "run_heuristic_checks",
    "build_record",
    "evaluate_records",
    "TEST_SET",
    "TestQuery",
    "by_id",
    "categories",
]
