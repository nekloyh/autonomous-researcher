// Mirrors app/state.py — keep in sync if backend types change.

export type TaskStatus = "pending" | "running" | "done" | "failed";

export interface SubTask {
  id: string;
  question: string;
  rationale: string;
  dependencies: string[];
  status?: TaskStatus;
  cell_id?: string;
  entity?: string;
  dimension?: string;
  target_queries?: string[];
  required_evidence?: number;
  success_criteria?: string[];
  allow_insufficient_data?: boolean;
}

export interface ToolCallCounts {
  web_search: number;
  fetch_url: number;
  vector_search: number;
  python_exec: number;
}

export interface Finding {
  task_id: string;
  sub_question?: string;
  answer?: string;
  content: string;
  claims?: Claim[];
  sources: string[];
  gaps?: string[];
  confidence: number;
  source_quality?: number;
  tool_calls: number;
  source_count?: number;
  // UI-only enrichment for compact live cards.
  excerpt?: string;
  tools?: ToolCallCounts;
}

export interface Claim {
  statement: string;
  source_url: string;
  snippet: string;
  confidence: number;
  source_domain?: string;
  source_type?: string;
  source_policy_tier?: "blocked" | "preferred" | "allowed";
  evidence_years?: string[];
  raw_snippet?: string;
  attributed_entities?: string[];
  validation_status?: "valid" | "low_confidence" | "dropped";
  validation_warnings?: string[];
  cell_id?: string;
  entity?: string;
  dimension?: string;
  evidence_type?: string;
  document_section?: string;
  page_or_chunk?: string;
}

export interface ResearchGap {
  question: string;
  origin_task_id?: string;
  reason?: string;
  priority?: "high" | "medium" | "low";
}

export interface SourceCandidate {
  url: string;
  title?: string;
  domain?: string;
  source_type?: "official" | "reputable_media" | "database" | "generic" | "unknown";
  source_policy_tier?: "blocked" | "preferred" | "allowed";
  year_status?: "matched" | "unknown" | "mismatch";
  rank_score?: number;
  assigned_task_ids?: string[];
  assigned_cell_ids?: string[];
}

export interface CriticScores {
  completeness: number;
  evidence: number;
  depth: number;
  accuracy: number;
  structure: number;
}

export interface Critique {
  action?: "finalize" | "research_gaps" | "replan";
  is_complete: boolean;
  quality_score: number;
  missing_info: string[];
  gaps?: ResearchGap[];
  factual_errors?: string[];
  unsupported_claims?: string[];
  conflicting_claims?: string[];
  suggestions: string[];
  // UI-only:
  scores?: CriticScores;
  threshold?: number;
}

export interface Citation {
  n: number;
  url: string;
  title: string;
  domain: string;
}

export type StageKey = "planner" | "researchers" | "synthesizer" | "critic";

export interface StageState {
  status: TaskStatus;
  elapsed: number | null;
  tokens: number | null;
}

export type Phase = "idle" | "running" | "done";

// SSE event payloads from /research/stream (see app/api/server.py:_summarize).
export type SSEUpdate =
  | {
      node: "planner" | "replan" | "gap_planner";
      plan_size: number;
      iteration: number;
      tasks: SubTask[];
      tokens: number;
    }
  | { node: "source_broker"; source_candidates: SourceCandidate[]; tool_calls: number }
  | {
      node: "researcher";
      task_id: string;
      sources: number;
      confidence: number;
      tool_calls: number;
      claims_count: number;
      excerpt: string;
      tokens: number;
    }
  | { node: "synthesizer"; draft_words: number; citations: number; tokens: number }
  | {
      node: "critic";
      action?: "finalize" | "research_gaps" | "replan";
      score: number;
      is_complete: boolean;
      missing: string[];
      gaps?: ResearchGap[];
      factual_errors?: string[];
      suggestions?: string[];
      tokens: number;
    }
  | {
      node: "finalize";
      final_words: number;
      quality_status?: "verified" | "partial" | "unverified";
      quality_warnings?: string[];
      run_summary_path?: string;
    };

export type SSEEvent =
  | { event: "start"; data: { session_id: string } }
  | { event: "update"; data: SSEUpdate }
  | {
      event: "done";
      data: {
        session_id: string;
        final_report: string;
        citations: string[];
        quality_status?: "verified" | "partial" | "unverified";
        quality_warnings?: string[];
        gaps?: ResearchGap[];
        run_summary_path?: string;
      };
    }
  | { event: "error"; data: { error: string } };
