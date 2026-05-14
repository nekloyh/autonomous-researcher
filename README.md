---
title: Autonomous Researcher
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.40.0"
app_file: app.py
pinned: false
---

# Autonomous Researcher

Multi-agent deep-research system. Decomposes a question into independent
sub-tasks, runs them in parallel through a ReAct researcher with four tools
(`web_search`, `fetch_url`, `vector_search`, `python_exec`), synthesizes a
cited markdown report, and uses a critic from a *different* model family to
trigger replans until the report meets a quality bar.

## Architecture

```
                ┌──────────┐
                │  START   │
                └────┬─────┘
                     ▼
                ┌──────────┐
        ┌───────│ Planner  │──────┐ (replan)
        │       └────┬─────┘      │
   fan-out (Send)    ▼            │
        │       ┌──────────┐      │
        ├──────▶│Researcher│──┐   │
        ├──────▶│Researcher│──┤   │
        ├──────▶│Researcher│──┤   │
        │       └──────────┘  │   │
        ▼                     ▼   │
   ┌────────────┐       ┌─────────┴──┐
   │Synthesizer │──────▶│   Critic   │
   └────────────┘       └─────┬──────┘
                              │
                              ▼
                         ┌─────────┐
                         │Finalize │──▶ END
                         └─────────┘
```

- **Planner** (Groq · `gemma2-9b-it`): decomposes user query into ≤ 5 SubTasks with
  Pydantic-validated structured output.
- **Researcher** (Groq · `llama-3.1-8b-instant` + 4 tools): runs a ReAct loop per
  SubTask, bounded to `REACT_MAX_STEPS` calls. Findings are merged via
  `Annotated[list, add]` reducers so parallel writes don't clobber each other.
- **Synthesizer** (Groq · `llama-3.3-70b-versatile`): aggregates findings into a
  markdown report. Citations are built deterministically from per-claim
  source URLs — only URLs the researcher actually fetched and tied to an atomic
  claim survive (defends against URL hallucination at the statement level).
- **Critic** (Google · `gemini-2.0-flash` — *different model family*): scores the
  draft on a 5-dim rubric (completeness, evidence, depth, accuracy, structure)
  and decides replan vs finalize. Threshold loosens per iteration to prevent
  infinite loops; `MAX_ITERATIONS` is a hard stop.

## Quickstart

```bash
# 1. Install
uv sync --all-extras

# 2. Set up secrets (Groq, Gemini, Tavily, optional LangSmith)
cp .env.example .env
$EDITOR .env

# Development mode is the default and uses deterministic local stubs, so CLI/API/UI
# runs do not spend LLM or search quota. Set APP_MODE=production to use real providers.

# 3. Start Qdrant + ingest seed corpus
docker compose up -d qdrant
uv run python scripts/ingest_corpus.py

# 4. Run a query (CLI)
uv run python main.py --query "Compare AI strategies of VNG and FPT in 2024" --stream

# 5. Or run the Streamlit UI
uv run streamlit run ui/streamlit_app.py

# 6. Or run the FastAPI server
uv run uvicorn app.api.server:app --reload --port 8000
```

You'll also need:
- **Ollama** running locally for embeddings (`ollama serve`, then
  `ollama pull nomic-embed-text`). Skip this if you swap embeddings via
  `app/config.py`.
- API keys for Groq, Google AI Studio, Tavily.

Production note: the hybrid `vector_search` reranker uses fastembed's
`Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder and downloads it on first use.
Preload it during setup with:

```bash
uv run python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"
```

## Tech stack

| Layer | Pick |
|---|---|
| Orchestration | LangGraph 0.2+ (StateGraph + `Send` fan-out + SqliteSaver) |
| Planner | Groq `gemma2-9b-it` (structured output) |
| Researcher | Groq `llama-3.1-8b-instant` (ReAct, fast tool-calling) |
| Synthesizer | Groq `llama-3.3-70b-versatile` (long-form report) |
| Critic | Google `gemini-2.0-flash` (different model family by design) |
| Embeddings | Ollama `nomic-embed-text` (local) — swap to HF for HF Spaces |
| Vector store | Qdrant (Docker locally, Qdrant Cloud in prod) |
| Search | DDGS/DuckDuckGo (free, primary) → Tavily (fallback when DDG fails) |
| Scraping | requests + BeautifulSoup + markdownify |
| API | FastAPI + SSE (sse-starlette) + slowapi rate-limit |
| UI | Streamlit + React (Vite + Tailwind) |
| Observability | LangSmith (auto-traced when configured) |
| Eval | RAGAS (faithfulness, relevancy, precision, recall) + heuristic checks |

## Layout

```
app/
├── agents/        planner, researcher, synthesizer, critic
├── api/           FastAPI server (REST + SSE)
├── evaluation/    heuristics, RAGAS pipeline, 20-query benchmark set
├── memory/        SqliteSaver checkpoint + Qdrant semantic memory
├── prompts/       4 versioned prompts (logged to LangSmith)
├── tools/         web_search, fetch_url, vector_search, python_exec
├── config.py      LLM factories + env config
├── graph.py       LangGraph orchestration
└── state.py       AgentState + reducers
ui/streamlit_app.py
main.py            CLI entry
scripts/
├── ingest_corpus.py   Qdrant corpus ingestion
├── run_eval.py        Eval orchestrator (--ab for critic on/off)
└── test_setup.py      Smoke test for all 6 services
corpus/            seed .md docs
tests/             pytest suite (29 tests)
```

## Running the eval harness

```bash
# 5-query smoke
uv run python scripts/run_eval.py --queries 5

# Full A/B: critic OFF (1 iter) vs critic ON (3 iters) on all 20 queries
uv run python scripts/run_eval.py --ab
```

Outputs land in `evaluation_outputs/eval_<timestamp>.{json,md}` with per-query
heuristic scores and aggregate RAGAS metrics. Targets: faithfulness ≥ 0.85,
answer relevancy ≥ 0.90, ≥ +15% improvement from the critic loop.

## Deploying to HuggingFace Spaces

1. The repo root is already configured (`app.py`, HF metadata in this README).
2. Generate a Spaces-friendly requirements file:
   `uv export --no-dev --extra ui --format requirements-txt > requirements.txt`
3. Set Space secrets: `GROQ_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`,
   `HF_SPACES=1`, and (optional) `QDRANT_URL` + `QDRANT_API_KEY` if you want
   the corpus tool to work.
4. `HF_SPACES=1` swaps embeddings from Ollama → `sentence-transformers/all-MiniLM-L6-v2`
   automatically (Ollama isn't available on Spaces). Code in `app/config.get_embeddings()`.
5. If you don't set Qdrant, the agent just falls back to the other 3 tools —
   `vector_search` returns a friendly "no corpus" message.
6. Push to the HF git remote — auto-build takes ~3-5 minutes.

## Patterns worth pointing out

- **Cross-model critic** — Llama generates, Gemini judges. Same-model evaluation
  has a known bias toward its own outputs; rotating model families is cheap and
  measurable insurance.
- **Two-stage researcher (ReAct → structured claims)** — the ReAct loop is left
  alone to do tool calls in free text; a second LLM call distils the transcript
  into `list[Claim]` (statement, source_url, snippet, confidence). Drives every
  downstream defence below.
- **Claim-level citation whitelist** — only URLs that a researcher actually
  fetched AND tied to a claim are eligible. The synthesizer's citation map is
  built deterministically from claims; `[N]` tokens the LLM invents are
  stripped, surviving ones are renumbered, and `## Sources` is auto-generated.
  URLs the LLM tries to hallucinate never make it into the final report.
- **Critic sees evidence** — the critic receives a structured findings/claims
  summary alongside the draft, and emits an `unsupported_claims` list. The
  graph routes to replan when claims aren't backed by evidence, not only when
  the rubric score is low.
- **Hybrid retrieval + local rerank** — `vector_search` runs dense (Qdrant) +
  BM25 (in-memory, pure-Python) in parallel, fuses with Reciprocal Rank Fusion,
  and reranks with a local fastembed cross-encoder. No paid rerank API.
- **Reducer-merged parallel writes** — `Annotated[list, add]` on `findings` and
  `critiques`, plus `Annotated[int, add]` on `total_tokens_used`, means
  LangGraph `Send`s converge cleanly without a manual fan-in node.
- **Threshold decay** — the critic's "is this good enough?" bar drops with each
  iteration (0.85 → 0.75 → 0.65). Combined with a hard `MAX_ITERATIONS` stop
  (passed via state, not env), this prevents the system from spinning on hard
  queries.
- **Sandboxed `python_exec`** — AST allowlist (modules + names), no `os`/`sys`/
  `open`/dunder access, restricted builtins. Math/stats only.
- **Untrusted-content firewall** — `fetch_url` wraps every webpage with an
  UNTRUSTED-CONTENT banner and redacts common prompt-injection patterns
  (`ignore previous instructions`, `you are now…`, role tags).
- **Tool fail-soft** — every tool returns an error string instead of raising,
  so the ReAct loop sees the failure and can react to it.

## What's measured

The agent self-reports per-run in the report footer:

```
*Generated by autonomous-researcher · iterations=2 · tool_calls=7 ·
 tokens≈12450 · cost≈$0.0062 · duration=43.2s*
```

`scripts/run_eval.py --ab` produces RAGAS faithfulness / context_precision /
context_recall (when ground truth is provided) + heuristic pass rate
(citations, sources section, claim-vs-snippet Jaccard, dead links, etc.) for
critic-off vs critic-on configs, side-by-side.

## Notes & caveats

- **Free-tier rate limits matter**: Groq is 30 RPM / 14.4k req/day, Tavily 1k
  searches/month. The `web_search` tool caches results; eval runs are I/O-heavy.
- **`vector_search` degrades gracefully** if Qdrant is down — the agent simply
  uses the other three tools.
- **CLI memory caching**: `main.py` consults `SemanticMemory` (Qdrant
  `past_queries` collection) before kicking off a fresh run. Pass `--no-memory`
  to bypass.

## Tests

```bash
uv run ruff check app/ tests/
uv run pytest -v
```

Current: **49 passing** (graph routing, sandbox, injection redaction, web
cache TTL, hybrid retrieval + RRF, synthesizer citation map, heuristics
including offline faithfulness, API).

## What I'd do next

- **Multi-query expansion / HyDE** in the researcher — one rewrite usually
  doubles recall on hard queries; deferred because each rewrite costs another
  tool call (free-tier sensitive).
- **Source-tier scoring** — weight a `.gov`/`.edu`/major-news domain higher
  than a forum post when scoring claim confidence.
- **Async tool wrappers** — `web_search`/`fetch_url` are sync and block the
  thread inside parallel `Send` fan-out; async would cut wall-clock latency.
- **Per-session budget circuit breaker** — hard cap on tokens/$ per run for
  cases where a query goes pathological.
- **Stale-cache invalidation in `SemanticMemory`** — currently TTL 7d with no
  keyword-based eviction; time-sensitive cached answers can mislead.
- **Fine-tuned planner** — `gemma2-9b-it` occasionally produces non-atomic
  sub-tasks; a small fine-tune on a few hundred good plans would help.
