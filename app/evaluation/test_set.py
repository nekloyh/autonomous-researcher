"""Benchmark queries for evaluating the autonomous researcher."""
from __future__ import annotations

from typing import TypedDict


class TestQuery(TypedDict, total=False):
    id: str
    category: str
    query: str
    required_facts: list[str]
    min_sources: int
    ground_truth: str


TEST_SET: list[TestQuery] = [
    # --- comparison ---
    {
        "id": "cmp_01",
        "category": "comparison",
        "query": "Compare AI strategies of VNG and FPT in 2024 — focus on products shipped, partnerships, and target customers.",
        "required_facts": ["VNG", "FPT", "2024"],
        "min_sources": 3,
    },
    {
        "id": "cmp_02",
        "category": "comparison",
        "query": "Compare LangGraph vs CrewAI for building production multi-agent systems. When would you pick which?",
        "required_facts": ["LangGraph", "CrewAI"],
        "min_sources": 3,
        "ground_truth": (
            "LangGraph models agent flow as an explicit state graph with cycles, "
            "checkpoints, and conditional edges — fits production systems that need "
            "deterministic replay and human-in-the-loop. CrewAI is role/task based "
            "with simpler abstractions and faster to start; pick it for "
            "prototypes or small autonomous crews where flow control is less critical."
        ),
    },
    {
        "id": "cmp_03",
        "category": "comparison",
        "query": "MoMo vs ZaloPay vs VNPay — market share, MAU, key product differentiators in 2024.",
        "required_facts": ["MoMo", "ZaloPay", "VNPay"],
        "min_sources": 3,
    },
    {
        "id": "cmp_04",
        "category": "comparison",
        "query": "Compare Qdrant, Milvus, and pgvector for production RAG: performance, ops, and cost.",
        "required_facts": ["Qdrant", "Milvus"],
        "min_sources": 3,
        "ground_truth": (
            "Qdrant: Rust-based, easy single-node deploy, good filtered search; "
            "Milvus: scale-out cluster mode, GPU options, heavier ops surface; "
            "pgvector: lives inside Postgres so no extra service, good when "
            "throughput < ~10M vectors and you already run Postgres."
        ),
    },
    {
        "id": "cmp_05",
        "category": "comparison",
        "query": "Compare Llama 3.3 70B and Qwen 2.5 72B for Vietnamese-language tasks in 2024.",
        "required_facts": ["Llama", "Qwen"],
        "min_sources": 2,
        "ground_truth": (
            "Both are strong multilingual 70-72B models with permissive licenses. "
            "Qwen 2.5 72B tends to outperform Llama 3.3 70B on Asian-language "
            "benchmarks (including Vietnamese) due to heavier Chinese/SEA training "
            "data, while Llama 3.3 has broader tooling/quantization support and is "
            "more widely served by inference providers like Groq."
        ),
    },
    # --- numerical ---
    {
        "id": "num_01",
        "category": "numerical",
        "query": "What was VNG Corporation's reported Q3 2024 revenue and YoY growth?",
        "required_facts": ["VNG", "Q3 2024"],
        "min_sources": 2,
        "ground_truth": (
            "VNG's Q3 2024 net revenue was reported around 2,300 billion VND, "
            "growing roughly 10-15% year over year compared to Q3 2023."
        ),
    },
    {
        "id": "num_02",
        "category": "numerical",
        "query": "How many monthly active users did MoMo report in 2024, and how does that compare to 2023?",
        "required_facts": ["MoMo", "MAU"],
        "min_sources": 2,
    },
    {
        "id": "num_03",
        "category": "numerical",
        "query": "What is the Tavily API free tier monthly search limit, and what does the 'basic' search depth cost in API credits?",
        "required_facts": ["Tavily"],
        "min_sources": 2,
        "ground_truth": (
            "Tavily's free tier provides 1,000 API credits per month. A `basic` "
            "depth search costs 1 credit per call; `advanced` depth costs 2."
        ),
    },
    {
        "id": "num_04",
        "category": "numerical",
        "query": "What are Groq's free-tier rate limits (RPM, TPM, daily) for the llama-3.3-70b-versatile model?",
        "required_facts": ["Groq", "llama-3.3-70b"],
        "min_sources": 2,
        "ground_truth": (
            "Groq free tier for llama-3.3-70b-versatile is around 30 requests/min "
            "and ~14,400 requests/day, with a TPM cap on the order of 6,000 tokens "
            "per minute. Exact numbers shift; check the live Groq console."
        ),
    },
    {
        "id": "num_05",
        "category": "numerical",
        "query": "Estimated COGS reduction from using Groq vs OpenAI for a 1M-token/day chat workload in 2024.",
        "required_facts": ["Groq", "OpenAI"],
        "min_sources": 2,
        "ground_truth": (
            "Groq Llama-3.3-70B is roughly $0.59/M input + $0.79/M output tokens; "
            "GPT-4o is ~$2.50/M input + $10/M output. A 1M-token/day workload "
            "(50/50 in/out) costs roughly $0.69/day on Groq vs ~$6.25/day on OpenAI "
            "— about 85-90% reduction at this mix and scale."
        ),
    },
    # --- analytical ---
    {
        "id": "ana_01",
        "category": "analytical",
        "query": "Analyze the impact of GenAI on the Vietnamese fintech sector in 2024. What use-cases dominate, what are the constraints?",
        "required_facts": ["GenAI", "Vietnam", "fintech"],
        "min_sources": 4,
    },
    {
        "id": "ana_02",
        "category": "analytical",
        "query": "Why did the Critic loop in agentic LLM systems become a popular pattern in 2024? What's the empirical evidence it works?",
        "required_facts": ["critic", "Reflexion"],
        "min_sources": 3,
        "ground_truth": (
            "The critic/reflection loop became popular because evaluator passes "
            "catch errors that single-shot generation misses. Reflexion (Shinn et al. "
            "2023) showed agents that critique their own outputs and retry improve "
            "task success rates on coding/reasoning benchmarks. Self-Refine and "
            "Constitutional AI made similar arguments. Empirical gains are usually "
            "+5-15 percentage points on hard tasks, with diminishing returns past "
            "~2-3 iterations."
        ),
    },
    {
        "id": "ana_03",
        "category": "analytical",
        "query": "What are the current best practices for evaluating retrieval-augmented generation (RAG) systems?",
        "required_facts": ["RAG", "evaluation"],
        "min_sources": 3,
    },
    {
        "id": "ana_04",
        "category": "analytical",
        "query": "Should a Vietnamese fintech startup train a custom LLM, fine-tune an open base, or just call APIs? Cost/risk tradeoffs in 2024.",
        "required_facts": ["fine-tune", "fintech"],
        "min_sources": 3,
    },
    {
        "id": "ana_05",
        "category": "analytical",
        "query": "How does prompt caching change the unit economics of agentic LLM applications in 2024?",
        "required_facts": ["prompt caching"],
        "min_sources": 2,
        "ground_truth": (
            "Prompt caching (Anthropic, OpenAI) lets providers store and reuse "
            "long system prompts/contexts, cutting cached-token cost by 50-90% and "
            "reducing TTFT. For agentic apps with stable system prompts and tool "
            "schemas, this turns the dominant cost from per-call inference into "
            "near-marginal — making multi-turn agents and long-context retrieval "
            "economically viable."
        ),
    },
    # --- multi-hop ---
    {
        "id": "mh_01",
        "category": "multi_hop",
        "query": "If VNG's revenue in Q3 2024 grew 12% YoY, calculate the implied Q3 2023 revenue and compare to the segment leader's known Q3 2023 revenue.",
        "required_facts": ["VNG"],
        "min_sources": 2,
    },
    {
        "id": "mh_02",
        "category": "multi_hop",
        "query": "Find the open-source LLMs that Vietnamese fintechs build on top of, then list the licenses of those LLMs.",
        "required_facts": ["Llama", "license"],
        "min_sources": 3,
    },
    {
        "id": "mh_03",
        "category": "multi_hop",
        "query": "Identify the top 3 multi-agent frameworks of 2024, then for each, list one production case-study.",
        "required_facts": ["LangGraph", "CrewAI"],
        "min_sources": 3,
        "ground_truth": (
            "LangGraph (LangChain) — used by Klarna and Elastic for agent workflows; "
            "CrewAI — adopted by enterprise teams for role-based crews; "
            "Microsoft AutoGen — research and Microsoft's own products such as Office "
            "Copilot prototypes. Production case studies vary by year; LangGraph and "
            "CrewAI dominate enterprise mindshare."
        ),
    },
    {
        "id": "mh_04",
        "category": "multi_hop",
        "query": "Find the most popular embedding model for Vietnamese text in 2024 and report its dimensionality and license.",
        "required_facts": ["embedding", "Vietnamese"],
        "min_sources": 2,
        "ground_truth": (
            "Multilingual general-purpose models dominate: BAAI/bge-m3 (1024 dim, "
            "MIT) and intfloat/multilingual-e5-large (1024 dim, MIT) are common "
            "choices for Vietnamese. nomic-embed-text (768 dim, Apache-2.0) is also "
            "widely used for local Ollama setups."
        ),
    },
    {
        "id": "mh_05",
        "category": "multi_hop",
        "query": "List the AI tools used inside MoMo's customer support, then identify which open-source LLM (if any) underpins each.",
        "required_facts": ["MoMo"],
        "min_sources": 3,
    },
]


def by_id(tid: str) -> TestQuery | None:
    return next((t for t in TEST_SET if t["id"] == tid), None)


def categories() -> set[str]:
    return {t["category"] for t in TEST_SET}
