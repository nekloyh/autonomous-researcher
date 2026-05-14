import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# Runtime mode
APP_MODE = os.getenv("APP_MODE", "development").strip().lower()
if APP_MODE not in {"development", "production"}:
    raise ValueError("APP_MODE must be either 'development' or 'production'")


def is_development() -> bool:
    return APP_MODE == "development"


def _env_required_in_production(name: str) -> str | None:
    value = os.getenv(name)
    if APP_MODE == "production" and not value:
        raise RuntimeError(f"{name} is required when APP_MODE=production")
    return value


# Env vars
GROQ_API_KEY = _env_required_in_production("GROQ_API_KEY")
GOOGLE_API_KEY = _env_required_in_production("GOOGLE_API_KEY")
TAVILY_API_KEY = _env_required_in_production("TAVILY_API_KEY")
GROQ_API_KEYS = os.getenv("GROQ_API_KEYS", GROQ_API_KEY or "")
GOOGLE_API_KEYS = os.getenv("GOOGLE_API_KEYS", GOOGLE_API_KEY or "")
TAVILY_API_KEYS = os.getenv("TAVILY_API_KEYS", TAVILY_API_KEY or "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Configs
MAX_ITERATIONS = int(os.getenv("MAX_RESEARCH_ITERATIONS", "3"))
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL_RESEARCHERS", "3"))
MAX_SUBTASKS = int(os.getenv("MAX_SUBTASKS", "5"))
MAX_GAP_ROUNDS = int(os.getenv("MAX_GAP_ROUNDS", "1"))
MAX_SOURCES_PER_TASK = int(os.getenv("MAX_SOURCES_PER_TASK", "4"))
SOURCE_BROKER_SEARCH_RESULTS = int(os.getenv("SOURCE_BROKER_SEARCH_RESULTS", "8"))
REACT_MAX_STEPS = 6  # Max tool calls per researcher
SEARCH_PROVIDER_ORDER = [
    p.strip().lower()
    for p in os.getenv("SEARCH_PROVIDER_ORDER", "tavily,ddg").split(",")
    if p.strip()
]


def _current_groq_key() -> str | None:
    from app.provider_rotation import GROQ_KEYS

    return GROQ_KEYS.current() or None


def _current_google_key() -> str | None:
    from app.provider_rotation import GOOGLE_KEYS

    return GOOGLE_KEYS.current() or None

# Model factory
def get_planner_llm():
    """Main LLM for planning. Needs reasoning."""
    from langchain_groq import ChatGroq

    from app.provider_rotation import GROQ_PLANNER_MODELS

    return ChatGroq(
        model=GROQ_PLANNER_MODELS.current(),
        temperature=0.2,
        max_tokens=2048,
        api_key=_current_groq_key(),
        max_retries=1,
    )


def get_researcher_llm():
    """LLM for ReAct researcher. Needs function calling."""
    from langchain_groq import ChatGroq

    from app.provider_rotation import GROQ_RESEARCHER_MODELS

    return ChatGroq(
        model=GROQ_RESEARCHER_MODELS.current(),
        temperature=0.1,
        api_key=_current_groq_key(),
        max_retries=1,
    )


def get_extractor_llm():
    """LLM for evidence extraction from ReAct transcripts."""
    from langchain_groq import ChatGroq

    from app.provider_rotation import GROQ_EXTRACTOR_MODELS

    return ChatGroq(
        model=GROQ_EXTRACTOR_MODELS.current(),
        temperature=0.0,
        api_key=_current_groq_key(),
        max_retries=1,
    )


def get_synthesizer_llm():
    """LLM for synthesis. Needs long output."""
    from langchain_groq import ChatGroq

    from app.provider_rotation import GROQ_SYNTHESIZER_MODELS

    return ChatGroq(
        model=GROQ_SYNTHESIZER_MODELS.current(),
        temperature=0.3,
        max_tokens=8000,  # Long reports
        api_key=_current_groq_key(),
        max_retries=1,
    )


def get_critic_llm():
    """LLM for critique. MUST be different model family for objectivity."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    from app.provider_rotation import GOOGLE_CRITIC_MODELS

    return ChatGoogleGenerativeAI(
        model=GOOGLE_CRITIC_MODELS.current(),
        temperature=0,
        api_key=_current_google_key(),
        retries=1,
    )


@lru_cache
def get_embeddings():
    """Embeddings provider.

    Local Ollama (`nomic-embed-text`) by default. When `HF_SPACES=1` is set,
    swap to a CPU-friendly HuggingFace model (Ollama isn't available on HF
    Spaces).
    """
    if os.getenv("HF_SPACES", "").strip() == "1":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_HOST,
    )
