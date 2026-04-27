"""Streamlit UI for autonomous-researcher.

Run: uv run streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

import streamlit as st

from app.graph import get_graph
from app.memory.checkpointer import get_checkpointer
from app.state import AgentState

PRESETS = [
    "So sánh chiến lược AI giữa VNG và FPT trong năm 2024",
    "Tác động của GenAI lên ngành fintech Việt Nam 2024",
    "What are the main differences between LangGraph and CrewAI?",
    "MoMo's competitive position vs ZaloPay and VNPay in 2024",
    "Top 3 vector databases for production RAG in 2025",
]


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


async def _run(query: str, session_id: str, slots: dict[str, Any]):
    graph = get_graph(checkpointer=get_checkpointer())
    config = {"configurable": {"thread_id": session_id}}

    plan_lines: list[str] = []
    findings_seen: dict[str, dict] = {}
    draft = ""
    citations: list[str] = []
    critiques_md: list[str] = []
    final = ""

    async for event in graph.astream(
        _initial_state(query, session_id), config=config, stream_mode="updates"
    ):
        for node, update in event.items():
            if node in ("planner", "replan"):
                tasks = update.get("plan") or []
                plan_lines = [
                    f"- **{t['id']}** — {t['question']}" for t in tasks
                ]
                slots["plan"].markdown(
                    f"**Iteration {update.get('current_iteration')}**\n\n"
                    + "\n".join(plan_lines)
                )
            elif node == "researcher":
                f = (update.get("findings") or [{}])[0]
                findings_seen[f.get("task_id", "?")] = f
                lines = [
                    f"### {fid}\n"
                    f"- confidence: `{ff.get('confidence', 0):.2f}`\n"
                    f"- sources: {len(ff.get('sources', []))}\n"
                    f"- excerpt: {(ff.get('content') or '')[:300]}…"
                    for fid, ff in findings_seen.items()
                ]
                slots["researchers"].markdown("\n\n".join(lines))
            elif node == "synthesizer":
                draft = update.get("draft_report") or draft
                citations = update.get("citations") or citations
                slots["synthesis"].markdown(draft)
            elif node == "critic":
                c = (update.get("critiques") or [{}])[-1]
                critiques_md.append(
                    f"- iter {len(critiques_md) + 1}: score=**{c.get('quality_score', 0):.2f}** · "
                    f"complete={c.get('is_complete')} · missing={len(c.get('missing_info') or [])}"
                )
                slots["critic"].markdown("\n".join(critiques_md))
            elif node == "finalize":
                final = update.get("final_report") or final
                slots["final"].markdown(final)
    return {"final": final, "citations": citations}


def main() -> None:
    st.set_page_config(page_title="Autonomous Researcher", page_icon="🔬", layout="wide")
    st.title("🔬 Autonomous Researcher")
    st.caption("Multi-agent deep research · Planner → Researchers → Synthesizer → Critic")

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("Quick start")
        for p in PRESETS:
            if st.button(p, use_container_width=True, key=f"preset_{hash(p)}"):
                st.session_state.preset = p
        st.divider()
        st.header("Session stats")
        st.metric("Past queries", len(st.session_state.history))

    initial_query = st.session_state.pop("preset", "")
    query = st.text_area(
        "Research question", value=initial_query, height=100, placeholder="Ask anything researchable…"
    )

    cols = st.columns([1, 1, 6])
    go = cols[0].button("🚀 Research", type="primary", use_container_width=True)
    cols[1].button("Clear", on_click=lambda: st.session_state.clear(), use_container_width=True)

    if go and query.strip():
        session_id = str(uuid.uuid4())[:8]
        st.session_state.history.append({"query": query, "session_id": session_id})

        with st.expander("📋 Plan", expanded=True):
            plan_slot = st.empty()
        with st.expander("🔍 Researchers", expanded=True):
            researchers_slot = st.empty()
        with st.expander("📝 Synthesis (draft)", expanded=False):
            synthesis_slot = st.empty()
        with st.expander("🧐 Critic", expanded=False):
            critic_slot = st.empty()

        st.divider()
        st.subheader("📄 Final report")
        final_slot = st.empty()

        slots = {
            "plan": plan_slot,
            "researchers": researchers_slot,
            "synthesis": synthesis_slot,
            "critic": critic_slot,
            "final": final_slot,
        }

        try:
            with st.spinner("Running agent…"):
                result = asyncio.run(_run(query.strip(), session_id, slots))
        except Exception as e:
            st.error(f"Run failed: {type(e).__name__}: {e}")
            return

        if result["final"]:
            st.download_button(
                "⬇️ Download report (.md)",
                data=result["final"],
                file_name=f"{session_id}.md",
                mime="text/markdown",
            )
            if result["citations"]:
                with st.expander(f"Citations ({len(result['citations'])})"):
                    for i, c in enumerate(result["citations"], 1):
                        st.markdown(f"[{i}] {c}")


if __name__ == "__main__":
    main()
