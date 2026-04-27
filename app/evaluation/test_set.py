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
    },
    {
        "id": "cmp_05",
        "category": "comparison",
        "query": "Compare Llama 3.3 70B and Qwen 2.5 72B for Vietnamese-language tasks in 2024.",
        "required_facts": ["Llama", "Qwen"],
        "min_sources": 2,
    },
    # --- numerical ---
    {
        "id": "num_01",
        "category": "numerical",
        "query": "What was VNG Corporation's reported Q3 2024 revenue and YoY growth?",
        "required_facts": ["VNG", "Q3 2024"],
        "min_sources": 2,
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
    },
    {
        "id": "num_04",
        "category": "numerical",
        "query": "What are Groq's free-tier rate limits (RPM, TPM, daily) for the llama-3.3-70b-versatile model?",
        "required_facts": ["Groq", "llama-3.3-70b"],
        "min_sources": 2,
    },
    {
        "id": "num_05",
        "category": "numerical",
        "query": "Estimated COGS reduction from using Groq vs OpenAI for a 1M-token/day chat workload in 2024.",
        "required_facts": ["Groq", "OpenAI"],
        "min_sources": 2,
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
    },
    {
        "id": "mh_04",
        "category": "multi_hop",
        "query": "Find the most popular embedding model for Vietnamese text in 2024 and report its dimensionality and license.",
        "required_facts": ["embedding", "Vietnamese"],
        "min_sources": 2,
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
