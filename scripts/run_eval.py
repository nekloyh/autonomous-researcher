"""Evaluation orchestrator.

Runs the agent over the benchmark set, computes heuristics + RAGAS, and saves
a JSON report. Supports an A/B mode that compares critic-on vs critic-off.

Usage:
    uv run python scripts/run_eval.py [--queries N] [--ab] [--out PATH]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.evaluation import TEST_SET, build_record, evaluate_records, run_heuristic_checks
from app.graph import build_graph, reset_graph
from app.memory.checkpointer import get_checkpointer
from app.state import AgentState

OUT_DIR = Path("evaluation_outputs")
OUT_DIR.mkdir(exist_ok=True)


def _initial_state(query: str, session_id: str) -> AgentState:
    return {
        "user_query": query,
        "session_id": session_id,
        "started_at": datetime.now(),
        "plan": [],
        "current_iteration": 0,
        "max_iterations": 0,
        "findings": [],
        "draft_report": "",
        "critiques": [],
        "final_report": "",
        "citations": [],
        "total_tool_calls": 0,
        "total_tokens_used": 0,
        "errors": [],
    }


async def _run_one(query: str) -> dict[str, Any]:
    reset_graph()
    graph = build_graph(checkpointer=get_checkpointer())
    session_id = "eval_" + uuid.uuid4().hex[:6]
    config = {"configurable": {"thread_id": session_id}}
    final = await graph.ainvoke(_initial_state(query, session_id), config=config)
    return final


async def _run_set(queries: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(queries, 1):
        print(f"[{label}] {i}/{len(queries)} {q['id']} :: {q['query'][:80]}…", flush=True)
        try:
            final = await _run_one(q["query"])
        except Exception as e:
            print(f"  ! failed: {type(e).__name__}: {e}", flush=True)
            rows.append({"id": q["id"], "error": str(e)})
            continue
        heur = run_heuristic_checks(final)
        rec = build_record(
            q["query"], final.get("final_report", ""), final.get("findings", []),
            ground_truth=q.get("ground_truth"),
        )
        rows.append(
            {
                "id": q["id"],
                "category": q.get("category"),
                "query": q["query"],
                "final_report": final.get("final_report", ""),
                "citations": final.get("citations", []) or [],
                "iterations": final.get("current_iteration"),
                "tool_calls": final.get("total_tool_calls"),
                "heuristics": heur,
                "ragas_record": rec,
            }
        )
        print(
            f"  iter={final.get('current_iteration')} "
            f"cites={len(final.get('citations') or [])} "
            f"heur_pass_rate={heur['_summary']['pass_rate']:.2f}",
            flush=True,
        )
    return rows


def _ragas_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = [r["ragas_record"] for r in rows if "ragas_record" in r and r.get("final_report")]
    if not records:
        return {"skipped": "no successful runs"}
    return evaluate_records(records)


def _markdown_table(label: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"## {label}", "", "| id | category | iters | cites | pass_rate |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('id', '?')} | {r.get('category', '?')} | "
            f"{r.get('iterations', '?')} | {len(r.get('citations') or [])} | "
            f"{r.get('heuristics', {}).get('_summary', {}).get('pass_rate', 0):.2f} |"
        )
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> None:
    queries = TEST_SET[: args.queries] if args.queries else TEST_SET

    if args.ab:
        os.environ["MAX_RESEARCH_ITERATIONS"] = "1"
        rows_off = await _run_set(queries, "critic_off")
        os.environ["MAX_RESEARCH_ITERATIONS"] = "3"
        rows_on = await _run_set(queries, "critic_on")
        ragas_off = _ragas_summary(rows_off)
        ragas_on = _ragas_summary(rows_on)
        report = {
            "timestamp": datetime.now().isoformat(),
            "ab": True,
            "critic_off": {"runs": rows_off, "ragas": ragas_off},
            "critic_on": {"runs": rows_on, "ragas": ragas_on},
        }
    else:
        rows = await _run_set(queries, "single")
        ragas = _ragas_summary(rows)
        report = {
            "timestamp": datetime.now().isoformat(),
            "ab": False,
            "runs": rows,
            "ragas": ragas,
        }

    out_path = Path(args.out) if args.out else OUT_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nWrote {out_path}")

    md_path = out_path.with_suffix(".md")
    md = []
    if args.ab:
        md.append("# A/B Eval Report\n")
        md.append(_markdown_table("Critic OFF", report["critic_off"]["runs"]))
        md.append("")
        md.append(_markdown_table("Critic ON", report["critic_on"]["runs"]))
        md.append("\n## RAGAS")
        md.append(f"- OFF aggregate: `{report['critic_off']['ragas'].get('aggregate', {})}`")
        md.append(f"- ON aggregate:  `{report['critic_on']['ragas'].get('aggregate', {})}`")
    else:
        md.append("# Eval Report\n")
        md.append(_markdown_table("Runs", report["runs"]))
        md.append(f"\n## RAGAS aggregate\n`{report['ragas'].get('aggregate', {})}`")
    md_path.write_text("\n".join(md))
    print(f"Wrote {md_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", type=int, default=0, help="Limit number of queries (0 = all)")
    p.add_argument("--ab", action="store_true", help="A/B test critic-on vs critic-off")
    p.add_argument("--out", type=str, default=None, help="Output path for JSON report")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
