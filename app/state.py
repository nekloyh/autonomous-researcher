from datetime import datetime
from operator import add
from typing import Annotated, Literal, TypedDict


# Sub-structures
class SubTask(TypedDict):
    id: str                    # "task_1", "task_2", ...
    question: str              # "Tìm doanh thu Q3 VNG"
    rationale: str             # Tại sao task này cần thiết
    dependencies: list[str]    # IDs của tasks cần chạy trước
    status: Literal["pending", "running", "done", "failed"]

class Claim(TypedDict):
    statement: str             # Atomic factual claim (1 sentence)
    source_url: str            # URL backing this claim
    snippet: str               # ≤300 chars from the source supporting it
    confidence: float          # 0-1

class Finding(TypedDict, total=False):
    task_id: str
    content: str               # Narrative summary (kept for backward compat)
    claims: list[Claim]        # Structured atomic facts (preferred)
    sources: list[str]         # URLs
    confidence: float          # 0-1
    tool_calls: int            # Số tool calls đã dùng

class Critique(TypedDict):
    is_complete: bool
    quality_score: float       # 0-1
    missing_info: list[str]    # Dạng questions cần research thêm
    factual_errors: list[str]
    suggestions: list[str]

# Main state
class AgentState(TypedDict):
    # Input
    user_query: str
    session_id: str
    started_at: datetime

    # Planning
    plan: list[SubTask]
    current_iteration: int
    max_iterations: int         # default = 3

    # Execution
    findings: Annotated[list[Finding], add]  # Append-only

    # Synthesis
    draft_report: str

    # Reflection
    critiques: Annotated[list[Critique], add]

    # Output
    final_report: str
    citations: list[str]

    # Meta
    total_tool_calls: int
    total_tokens_used: Annotated[int, add]
    errors: Annotated[list[str], add]

# Pre-researcher state
class ResearcherState(TypedDict):
    task: SubTask
    user_query: str  # Original query for context
    session_id: str
