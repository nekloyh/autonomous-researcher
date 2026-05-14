"""FastAPI server: REST + SSE endpoints for the research agent."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from app.config import MAX_ITERATIONS, is_development
from app.graph import get_graph
from app.memory.checkpointer import get_checkpointer
from app.state import AgentState

API_VERSION = "0.1.0"

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="autonomous-researcher",
    version=API_VERSION,
    description="Multi-agent deep research API (Planner → Researchers → Synthesizer → Critic).",
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    print(
        f"{request.method} {request.url.path} -> {response.status_code} "
        f"({elapsed:.1f}ms)",
        flush=True,
    )
    return response


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    session_id: str | None = None


class ResearchResponse(BaseModel):
    session_id: str
    final_report: str
    citations: list[str]
    iterations: int
    total_tool_calls: int
    quality_status: str = "unverified"
    quality_warnings: list[str] = Field(default_factory=list)
    gaps: list[dict[str, Any]] = Field(default_factory=list)
    run_summary_path: str = ""


def _initial_state(query: str, session_id: str) -> AgentState:
    return {
        "user_query": query,
        "session_id": session_id,
        "started_at": datetime.now(),
        "plan": [],
        "current_iteration": 0,
        "max_iterations": MAX_ITERATIONS,
        "gap_rounds": 0,
        "findings": [],
        "source_candidates": [],
        "draft_report": "",
        "critiques": [],
        "final_report": "",
        "citations": [],
        "quality_status": "unverified",
        "quality_warnings": [],
        "run_summary_path": "",
        "total_tool_calls": 0,
        "total_tokens_used": 0,
        "errors": [],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    from app.config import APP_MODE

    qdrant_ok = True
    try:
        from qdrant_client import QdrantClient

        from app.config import QDRANT_URL

        QdrantClient(url=QDRANT_URL).get_collections()
    except Exception:
        qdrant_ok = False
    return {"status": "ok", "version": API_VERSION, "mode": APP_MODE, "qdrant": qdrant_ok}


@app.post("/research", response_model=ResearchResponse)
@limiter.limit("5/minute")
async def research(request: Request, body: ResearchRequest):
    session_id = body.session_id or str(uuid.uuid4())[:8]
    graph = get_graph(checkpointer=None if is_development() else get_checkpointer())
    config = {"configurable": {"thread_id": session_id}}
    final = await graph.ainvoke(_initial_state(body.query, session_id), config=config)
    return ResearchResponse(
        session_id=session_id,
        final_report=final.get("final_report", ""),
        citations=final.get("citations", []) or [],
        iterations=final.get("current_iteration", 0),
        total_tool_calls=final.get("total_tool_calls", 0),
        quality_status=final.get("quality_status", "unverified"),
        quality_warnings=final.get("quality_warnings", []) or [],
        gaps=[
            gap
            for critique in final.get("critiques", []) or []
            for gap in (critique.get("gaps") or [])
        ],
        run_summary_path=final.get("run_summary_path", ""),
    )


def _summarize(node: str, update: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"node": node}
    if node in ("planner", "replan", "gap_planner"):
        plan = update.get("plan") or []
        out["plan_size"] = len(plan)
        out["iteration"] = update.get("current_iteration")
        out["tasks"] = [
            {"id": t.get("id"), "question": t.get("question"), "rationale": t.get("rationale")}
            for t in plan
        ]
        out["tokens"] = update.get("total_tokens_used") or 0
    elif node == "source_broker":
        sources = update.get("source_candidates") or []
        out["source_candidates"] = [
            {
                "url": s.get("url"),
                "title": s.get("title"),
                "domain": s.get("domain"),
                "source_type": s.get("source_type"),
                "rank_score": s.get("rank_score"),
                "assigned_task_ids": s.get("assigned_task_ids") or [],
            }
            for s in sources[:10]
        ]
        out["tool_calls"] = update.get("total_tool_calls") or 0
    elif node == "researcher":
        f = (update.get("findings") or [{}])[0]
        content = f.get("content") or ""
        out["task_id"] = f.get("task_id")
        out["sources"] = len(f.get("sources") or [])
        out["confidence"] = f.get("confidence")
        out["tool_calls"] = f.get("tool_calls")
        out["claims_count"] = len(f.get("claims") or [])
        out["excerpt"] = content[:300]
        out["tokens"] = update.get("total_tokens_used") or 0
    elif node == "synthesizer":
        out["draft_words"] = len((update.get("draft_report") or "").split())
        out["citations"] = len(update.get("citations") or [])
        out["tokens"] = update.get("total_tokens_used") or 0
    elif node == "critic":
        c = (update.get("critiques") or [{}])[-1]
        out["action"] = c.get("action")
        out["score"] = c.get("quality_score")
        out["is_complete"] = c.get("is_complete")
        out["missing"] = c.get("missing_info") or []
        out["gaps"] = c.get("gaps") or []
        out["factual_errors"] = c.get("factual_errors") or []
        out["suggestions"] = c.get("suggestions") or []
        out["tokens"] = update.get("total_tokens_used") or 0
    elif node == "finalize":
        final = update.get("final_report") or ""
        out["final_words"] = len(final.split())
        out["quality_status"] = update.get("quality_status")
        out["quality_warnings"] = update.get("quality_warnings") or []
        out["run_summary_path"] = update.get("run_summary_path", "")
    return out


@app.post("/research/stream")
@limiter.limit("5/minute")
async def research_stream(request: Request, body: ResearchRequest):
    session_id = body.session_id or str(uuid.uuid4())[:8]
    graph = get_graph(checkpointer=None if is_development() else get_checkpointer())
    config = {"configurable": {"thread_id": session_id}}

    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        yield {"event": "start", "data": json.dumps({"session_id": session_id})}
        try:
            async for event in graph.astream(
                _initial_state(body.query, session_id),
                config=config,
                stream_mode="updates",
            ):
                for node, update in event.items():
                    yield {
                        "event": "update",
                        "data": json.dumps(_summarize(node, update), default=str),
                    }
            final = (await graph.aget_state(config)).values
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "session_id": session_id,
                        "final_report": final.get("final_report", ""),
                        "citations": final.get("citations", []) or [],
                        "quality_status": final.get("quality_status", "unverified"),
                        "quality_warnings": final.get("quality_warnings", []) or [],
                        "gaps": [
                            gap
                            for critique in final.get("critiques", []) or []
                            for gap in (critique.get("gaps") or [])
                        ],
                        "run_summary_path": final.get("run_summary_path", ""),
                    },
                    default=str,
                ),
            }
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())
