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

- **Planner** (Groq · Llama 3.3 70B): decomposes user query into ≤ 5 SubTasks with
  Pydantic-validated structured output.
- **Researcher** (Groq · Llama 3.3 70B + 4 tools): runs a ReAct loop per SubTask,
  bounded to `REACT_MAX_STEPS` calls. Findings are merged via `Annotated[list, add]`
  reducers so parallel writes don't clobber each other.
- **Synthesizer** (Groq · Llama 3.3 70B): aggregates findings into a markdown
  report. Citations are **whitelisted** — only URLs that actually appeared in tool
  outputs survive (defends against URL hallucination).
- **Critic** (Google · Gemini 2.0 Flash — *different model family*): scores the
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

## Tech stack

| Layer | Pick |
|---|---|
| Orchestration | LangGraph 0.2+ (StateGraph + `Send` fan-out + SqliteSaver) |
| Generators | Groq Llama 3.3 70B (Planner / Researcher / Synthesizer) |
| Critic | Google Gemini 2.0 Flash (different model family by design) |
| Embeddings | Ollama nomic-embed-text (local) — swap to HF/Gemini for HF Spaces |
| Vector store | Qdrant (Docker locally, Qdrant Cloud in prod) |
| Search | Tavily (primary) → DuckDuckGo (fallback) |
| Scraping | requests + BeautifulSoup + markdownify |
| API | FastAPI + SSE (sse-starlette) + slowapi rate-limit |
| UI | Streamlit |
| Observability | LangSmith (auto-traced) |
| Eval | RAGAS (faithfulness, relevancy, precision, recall) + 6 heuristic checks |

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
   `QDRANT_URL`, `QDRANT_API_KEY`.
4. HF Spaces can't run Ollama. Either:
   - Use Qdrant Cloud + remote inference for embeddings, or
   - Switch `app/config.get_embeddings()` to `langchain_huggingface.HuggingFaceEmbeddings`
     (sentence-transformers / `all-MiniLM-L6-v2`).
5. Push to the HF git remote — auto-build takes ~3-5 minutes.

## Patterns worth pointing out

- **Cross-model critic** — Llama generates, Gemini judges. Same-model evaluation
  has a known bias toward its own outputs; rotating model families is cheap and
  measurable insurance.
- **Citation whitelist** — the synthesizer's output goes through a regex
  extractor that drops any URL that didn't actually appear in a tool output.
  Closes the most common hallucination vector for this kind of system.
- **Reducer-merged parallel writes** — `Annotated[list, add]` on `findings` and
  `critiques` means LangGraph `Send`s converge cleanly without a manual
  fan-in node.
- **Threshold decay** — the critic's "is this good enough?" bar drops with each
  iteration (0.85 → 0.75 → 0.65). Combined with a hard `MAX_ITERATIONS` stop,
  this prevents the system from spinning on hard queries.
- **Tool fail-soft** — every tool returns an error string instead of raising,
  so the ReAct loop sees the failure and can react to it.

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

Current: **29 passing** (graph routing, heuristics, tools, API).
