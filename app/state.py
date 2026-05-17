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
    cell_id: str
    entity: str
    dimension: str
    target_queries: list[str]
    required_evidence: int
    success_criteria: list[str]
    allow_insufficient_data: bool

class Claim(TypedDict):
    statement: str             # Atomic factual claim (1 sentence)
    source_url: str            # URL backing this claim
    snippet: str               # ≤300 chars from the source supporting it
    confidence: float          # 0-1
    source_domain: str
    source_type: str
    source_policy_tier: Literal["blocked", "preferred", "allowed"]
    evidence_years: list[str]
    raw_snippet: str
    attributed_entities: list[str]
    validation_status: Literal["valid", "low_confidence", "dropped"]
    validation_warnings: list[str]
    cell_id: str
    entity: str
    dimension: str
    evidence_type: str
    document_section: str
    page_or_chunk: str

class SourceCandidate(TypedDict, total=False):
    url: str
    canonical_url: str
    title: str
    snippet: str
    domain: str
    rank_score: float
    source_type: Literal["official", "reputable_media", "database", "generic", "unknown"]
    source_policy_tier: Literal["blocked", "preferred", "allowed"]
    year_status: Literal["matched", "unknown", "mismatch"]
    assigned_task_ids: list[str]
    assigned_cell_ids: list[str]

class ResearchGap(TypedDict, total=False):
    question: str
    origin_task_id: str
    reason: str
    priority: Literal["high", "medium", "low"]

class Finding(TypedDict, total=False):
    task_id: str
    sub_question: str
    answer: str
    content: str               # Narrative summary (kept for backward compat)
    claims: list[Claim]        # Structured atomic facts (preferred)
    sources: list[str]         # URLs
    gaps: list[str]
    confidence: float          # 0-1
    source_quality: float      # 0-1
    tool_calls: int            # Số tool calls đã dùng
    dropped_claims: list[Claim]
    validation_gaps: list[ResearchGap]
    researcher_error_status: Literal["none", "recovered", "unrecovered"]
    cell_id: str
    entity: str
    dimension: str

class Critique(TypedDict, total=False):
    action: Literal["finalize", "research_gaps", "replan"]
    is_complete: bool
    quality_score: float       # 0-1
    missing_info: list[str]    # Dạng questions cần research thêm
    gaps: list[ResearchGap]
    factual_errors: list[str]
    unsupported_claims: list[str]
    conflicting_claims: list[str]
    suggestions: list[str]

# Main state
class AgentState(TypedDict):
    # Input
    user_query: str
    session_id: str
    started_at: datetime

    # Planning
    plan: list[SubTask]
    research_plan: dict
    cell_coverage: list[dict]
    current_iteration: int
    max_iterations: int         # default = 3
    gap_rounds: int

    # Execution
    findings: Annotated[list[Finding], add]  # Append-only
    source_candidates: Annotated[list[SourceCandidate], add]

    # Synthesis
    draft_report: str

    # Reflection
    critiques: Annotated[list[Critique], add]

    # Output
    final_report: str
    citations: list[str]
    quality_status: Literal["verified", "partial", "unverified"]
    quality_warnings: list[str]
    run_summary_path: str

    # Meta
    total_tool_calls: Annotated[int, add]
    total_tokens_used: Annotated[int, add]
    errors: Annotated[list[str], add]

# Pre-researcher state
class ResearcherState(TypedDict):
    task: SubTask
    user_query: str  # Original query for context
    session_id: str
    assigned_sources: list[SourceCandidate]
