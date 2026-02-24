from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for the RAG Engineering Lab.

    Values are loaded from environment variables (or a .env file).
    pydantic-settings validates and coerces types at startup, so a missing
    required variable raises an error immediately rather than at call time.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # silently ignore env vars not declared here
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str

    llm_model: str = "claude-haiku-4-5-20251001"
    llm_temperature: float = 0.0

    # ── Embeddings ────────────────────────────────────────────────────────────
    # all-MiniLM-L6-v2: local, free, ~80 MB, 384-dimensional vectors.
    # Chosen because it is fast, well-benchmarked on semantic similarity tasks,
    # and requires no API key — important for a local-first dev environment.
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    max_retrieval_attempts: int = 3

    # ── Grounding ─────────────────────────────────────────────────────────────
    # The grounding node asks the LLM to score how well the answer is supported
    # by the cited chunks (0.0–1.0). Answers below this threshold get a warning
    # appended to the final response.
    grounding_threshold: float = 0.7

    # ── Storage ───────────────────────────────────────────────────────────────
    # Using Path objects (not raw strings) so callers can do path arithmetic
    # (e.g., uploads_dir / filename) without string concatenation.
    uploads_dir: Path = Path("data/uploads")
    chroma_dir: Path = Path("data/chroma")

    # ── LangSmith ─────────────────────────────────────────────────────────────
    # LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY are read automatically by the
    # LangSmith SDK from the environment — no code needed. We declare
    # LANGCHAIN_PROJECT here so it is validated and accessible as a typed
    # attribute if needed elsewhere (e.g., to display in the UI).
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "false"
    langchain_project: str = "rag-engineering-lab"


# Module-level singleton: import this everywhere instead of instantiating
# Settings() in multiple places. Constructed once at import time, so the
# .env file is read and validated exactly once per process.
settings = Settings()
