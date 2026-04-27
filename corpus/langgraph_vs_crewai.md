# LangGraph vs CrewAI — Framework Comparison Notes

**Source**: Internal research note (synthetic seed).
**Date**: 2024-12-10

Two of the most prominent multi-agent frameworks in 2024 are LangGraph (by LangChain) and CrewAI. They occupy adjacent but distinct niches.

## LangGraph

- **Paradigm**: explicit graph (nodes, edges, conditional edges). State is a typed dict with reducers.
- **Strengths**: full control of orchestration; deterministic; first-class checkpointing (resume mid-run); great for production where you need to debug each transition.
- **Weaknesses**: more boilerplate; conceptual learning curve (`Send`, `Annotated[list, add]`, etc.).
- **Patterns supported**: ReAct, Plan-and-Execute, Reflexion, fan-out/fan-in, hierarchical teams, human-in-the-loop.

## CrewAI

- **Paradigm**: role-based "crew" with agents, tasks, and a Process (sequential or hierarchical).
- **Strengths**: very ergonomic for rapid prototypes; minimal code; nice abstractions (Agent, Task, Crew).
- **Weaknesses**: harder to express arbitrary topologies (e.g., conditional replanning loops with critic-driven fan-out); less mature checkpointing.
- **Patterns supported**: best at "team of role-based agents collaborating on a goal".

## When to pick which

- **Production deep-research / agentic backends**: LangGraph wins because you usually need explicit DAGs, parallel fan-out, replan loops, and checkpoint resume.
- **Prototypes, POCs, demos**: CrewAI is faster to wire up.
- **Hybrid**: it's common to prototype in CrewAI, then migrate hot paths to LangGraph for control.

## Production considerations

- LangGraph integrates natively with LangSmith for observability — every node transition is traced.
- Both can use the same underlying LLMs (Groq, OpenAI, Gemini, Anthropic) and tool ecosystem (LangChain tools).
- For "deep research" style applications (Planner → parallel Researchers → Synthesizer → Critic), LangGraph's `Send` API and reducer-based state merging are notably better fits than CrewAI's hierarchical Process.
