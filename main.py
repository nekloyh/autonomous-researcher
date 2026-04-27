"""CLI entry point for autonomous-researcher.

Usage:
    uv run python main.py --query "Your research question here"
    uv run python main.py --query "..." --session-id my-session --stream
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.graph import get_graph
from app.memory.checkpointer import get_checkpointer
from app.memory.long_term import SemanticMemory
from app.state import AgentState

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)


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


def _summarize_update(node: str, update: dict[str, Any]) -> str:
    if node == "planner" or node == "replan":
        plan = update.get("plan") or []
        return f"plan with {len(plan)} sub-tasks (iter {update.get('current_iteration', '?')})"
    if node == "researcher":
        findings = update.get("findings") or []
        if findings:
            f = findings[0]
            return (
                f"task={f.get('task_id')} sources={len(f.get('sources', []))} "
                f"conf={f.get('confidence', 0):.2f}"
            )
    if node == "synthesizer":
        draft = update.get("draft_report") or ""
        return f"draft={len(draft.split())} words, citations={len(update.get('citations') or [])}"
    if node == "critic":
        critiques = update.get("critiques") or []
        if critiques:
            c = critiques[-1]
            return (
                f"score={c.get('quality_score', 0):.2f} complete={c.get('is_complete')} "
                f"missing={len(c.get('missing_info') or [])}"
            )
    if node == "finalize":
        final = update.get("final_report") or ""
        return f"final report = {len(final.split())} words"
    return ", ".join(update.keys())


async def _run(query: str, session_id: str, stream: bool, use_memory: bool) -> dict:
    if use_memory:
        mem = SemanticMemory()
        cached = mem.find_similar(query)
        if cached:
            print(f"[memory] hit (similarity {cached['score']:.2f}, age {cached['age_days']}d)")
            return {"final_report": cached["final_report"], "citations": cached.get("citations", [])}

    graph = get_graph(checkpointer=get_checkpointer())
    state = _initial_state(query, session_id)
    config = {"configurable": {"thread_id": session_id}}

    last_state: dict[str, Any] = {}
    async for event in graph.astream(state, config=config, stream_mode="updates"):
        for node, update in event.items():
            if stream:
                print(f"[{node}] {_summarize_update(node, update)}", flush=True)
            last_state.update(update)

    final = graph.get_state(config).values
    if use_memory and final.get("final_report"):
        SemanticMemory().store(
            query=query,
            final_report=final["final_report"],
            session_id=session_id,
            citations=final.get("citations", []),
        )
    return final


def main() -> int:
    p = argparse.ArgumentParser(description="Run the autonomous research agent")
    p.add_argument("--query", "-q", required=True, help="Research question")
    p.add_argument("--session-id", "-s", default=None, help="Resume / tag a session")
    p.add_argument("--stream", action="store_true", help="Print live node updates")
    p.add_argument(
        "--no-memory",
        action="store_true",
        help="Skip long-term semantic memory lookup/store",
    )
    args = p.parse_args()

    session_id = args.session_id or str(uuid.uuid4())[:8]
    print(f"Query: {args.query}")
    print(f"Session: {session_id}\n")

    result = asyncio.run(_run(args.query, session_id, args.stream, not args.no_memory))

    final = result.get("final_report") or "(no report generated)"
    citations = result.get("citations") or []

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(final)
    if citations:
        print("\nCITATIONS")
        print("-" * 70)
        for i, c in enumerate(citations, 1):
            print(f"[{i}] {c}")

    out_md = OUTPUTS / f"{session_id}.md"
    out_md.write_text(final, encoding="utf-8")
    out_json = OUTPUTS / f"{session_id}.json"
    out_json.write_text(
        json.dumps(
            {
                "query": args.query,
                "session_id": session_id,
                "final_report": final,
                "citations": citations,
                "iterations": result.get("current_iteration"),
                "total_tool_calls": result.get("total_tool_calls"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved → {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
