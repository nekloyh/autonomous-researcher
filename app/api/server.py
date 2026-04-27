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


@app.get("/health")
def health() -> dict[str, Any]:
    qdrant_ok = True
    try:
        from qdrant_client import QdrantClient

        from app.config import QDRANT_URL

        QdrantClient(url=QDRANT_URL).get_collections()
    except Exception:
        qdrant_ok = False
    return {"status": "ok", "version": API_VERSION, "qdrant": qdrant_ok}


@app.post("/research", response_model=ResearchResponse)
@limiter.limit("5/minute")
async def research(request: Request, body: ResearchRequest):
    session_id = body.session_id or str(uuid.uuid4())[:8]
    graph = get_graph(checkpointer=get_checkpointer())
    config = {"configurable": {"thread_id": session_id}}
    final = await graph.ainvoke(_initial_state(body.query, session_id), config=config)
    return ResearchResponse(
        session_id=session_id,
        final_report=final.get("final_report", ""),
        citations=final.get("citations", []) or [],
        iterations=final.get("current_iteration", 0),
        total_tool_calls=final.get("total_tool_calls", 0),
    )


def _summarize(node: str, update: dict[str, Any]) -> dict[str, Any]:
    out = {"node": node}
    if node in ("planner", "replan"):
        out["plan_size"] = len(update.get("plan") or [])
        out["iteration"] = update.get("current_iteration")
    elif node == "researcher":
        f = (update.get("findings") or [{}])[0]
        out["task_id"] = f.get("task_id")
        out["sources"] = len(f.get("sources", []))
        out["confidence"] = f.get("confidence")
    elif node == "synthesizer":
        out["draft_words"] = len((update.get("draft_report") or "").split())
        out["citations"] = len(update.get("citations") or [])
    elif node == "critic":
        c = (update.get("critiques") or [{}])[-1]
        out["score"] = c.get("quality_score")
        out["is_complete"] = c.get("is_complete")
        out["missing"] = len(c.get("missing_info") or [])
    elif node == "finalize":
        out["final_words"] = len((update.get("final_report") or "").split())
    return out


@app.post("/research/stream")
@limiter.limit("5/minute")
async def research_stream(request: Request, body: ResearchRequest):
    session_id = body.session_id or str(uuid.uuid4())[:8]
    graph = get_graph(checkpointer=get_checkpointer())
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
                    },
                    default=str,
                ),
            }
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())
