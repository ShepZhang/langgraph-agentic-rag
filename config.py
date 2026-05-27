"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - used before dependencies are installed.
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        """Fallback so config can be imported before dependencies are installed."""

        return False


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Typed runtime settings for the Agentic RAG MVP."""

    openai_api_key: str
    openai_base_url: str
    openai_model: str
    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_rewrite_attempts: int
    chroma_persist_dir: Path
    chroma_collection_name: str
    gradio_server_name: str
    gradio_server_port: int

    @property
    def has_llm_config(self) -> bool:
        """Return True when the required chat LLM settings are present."""

        return bool(self.openai_api_key and self.openai_model)

    def require_llm_config(self) -> None:
        """Raise a clear error if the chat LLM is not configured."""

        if not self.has_llm_config:
            raise RuntimeError(
                "Missing LLM configuration. Set OPENAI_API_KEY and OPENAI_MODEL "
                "in your environment or .env file before running Agentic RAG."
            )


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def get_settings() -> Settings:
    """Load settings from environment variables."""

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ).strip(),
        chunk_size=_get_int("CHUNK_SIZE", 800),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
        top_k=_get_int("TOP_K", 4),
        max_rewrite_attempts=_get_int("MAX_REWRITE_ATTEMPTS", 2),
        chroma_persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME",
            "agentic_rag_documents",
        ).strip(),
        gradio_server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1").strip(),
        gradio_server_port=_get_int("GRADIO_SERVER_PORT", 7860),
    )
