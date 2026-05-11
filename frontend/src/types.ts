// Mirrors app/state.py — keep in sync if backend types change.

export type TaskStatus = "pending" | "running" | "done" | "failed";

export interface SubTask {
  id: string;
  question: string;
  rationale: string;
  dependencies: string[];
  status?: TaskStatus;
}

export interface ToolCallCounts {
  web_search: number;
  fetch_url: number;
  vector_search: number;
  python_exec: number;
}

export interface Finding {
  task_id: string;
  content: string;
  sources: string[];
  confidence: number;
  tool_calls: number;
  // UI-only enrichment (mock/demo); the live SSE payload does not carry these.
  excerpt?: string;
  tools?: ToolCallCounts;
}

export interface CriticScores {
  completeness: number;
  evidence: number;
  depth: number;
  accuracy: number;
  structure: number;
}

export interface Critique {
  is_complete: boolean;
  quality_score: number;
  missing_info: string[];
  factual_errors?: string[];
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
  | { node: "planner" | "replan"; plan_size: number; iteration: number }
  | { node: "researcher"; task_id: string; sources: number; confidence: number }
  | { node: "synthesizer"; draft_words: number; citations: number }
  | { node: "critic"; score: number; is_complete: boolean; missing: number }
  | { node: "finalize"; final_words: number };

export type SSEEvent =
  | { event: "start"; data: { session_id: string } }
  | { event: "update"; data: SSEUpdate }
  | { event: "done"; data: { session_id: string; final_report: string; citations: string[] } }
  | { event: "error"; data: { error: string } };
