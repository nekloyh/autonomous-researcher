# GenAI Adoption in Vietnamese Fintech — 2024 Outlook

**Source**: Internal research note (synthetic seed).
**Date**: 2024-12-01

Vietnamese fintech firms accelerated GenAI adoption in 2024, focusing on three pragmatic use-cases rather than frontier model training: customer support automation, KYC/document understanding, and fraud detection.

## Adoption patterns

- **Build vs. buy**: Almost all top-10 fintechs (MoMo, ZaloPay, VNPay, Tima, Finhay, etc.) build *on top of* open-source LLMs (primarily Llama 3.1/3.3 and Qwen 2.5) fine-tuned on Vietnamese conversational data, rather than training from scratch.
- **Vector search**: Qdrant and Milvus are the dominant choices; pgvector is used by smaller teams already on Postgres.
- **Inference cost**: Most teams cite USD 0.05–0.20 per ticket served via LLM-powered chat as the affordability threshold; Groq and Together AI are commonly used for inference.

## Regulatory context

The State Bank of Vietnam (SBV) issued draft guidance in Q3 2024 requiring fintech firms using AI for credit-scoring or fraud-decisioning to (a) keep human-in-the-loop reviews above a threshold of impact, (b) document model lineage, and (c) provide explanations on adverse decisions. This mirrors EU AI Act high-risk requirements but with a more lenient "guidance" enforcement posture in 2024.

## Outlook 2025

- Multi-agent / "deep research" assistants for financial analysts (à la OpenAI Deep Research) are an emerging interest, with several banks running internal POCs.
- Voice-first GenAI for low-literacy customer segments is a differentiator; FPT.AI and Zalo are leading.
- Expect at least one Vietnamese fintech to ship an *autonomous* customer-onboarding agent (paperwork → KYC → first product) by end of 2025.
