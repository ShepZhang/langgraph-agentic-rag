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

    llm_provider: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    ollama_base_url: str
    ollama_model: str
    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_retry_count: int
    temperature: float
    chroma_persist_dir: Path
    chroma_collection_name: str
    gradio_server_name: str
    gradio_server_port: int

    def __post_init__(self) -> None:
        """Validate settings early so runtime failures are explicit."""

        if self.llm_provider not in {"openai_compatible", "ollama"}:
            raise ValueError(
                "LLM_PROVIDER must be either 'openai_compatible' or 'ollama'"
            )
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP must be greater than or equal to 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be greater than 0")
        if self.max_retry_count < 0:
            raise ValueError("MAX_RETRY_COUNT must be greater than or equal to 0")
        if not 0 <= self.temperature <= 2:
            raise ValueError(
                "LLM_TEMPERATURE/OPENAI_TEMPERATURE must be between 0 and 2"
            )
        if not str(self.chroma_persist_dir).strip():
            raise ValueError("CHROMA_PERSIST_DIR must not be empty")
        if not self.chroma_collection_name:
            raise ValueError("CHROMA_COLLECTION_NAME must not be empty")

    @property
    def max_rewrite_attempts(self) -> int:
        """Backward-compatible alias for older UI/docs code."""

        return self.max_retry_count

    @property
    def has_llm_config(self) -> bool:
        """Return True when the required chat LLM settings are present."""

        if self.llm_provider == "ollama":
            return bool(self.ollama_base_url and self.ollama_model)
        return bool(self.openai_api_key and self.openai_base_url and self.openai_model)

    @property
    def effective_llm_api_key(self) -> str:
        """API key passed to the chat model client."""

        if self.llm_provider == "ollama":
            return "ollama"
        return self.openai_api_key

    @property
    def effective_llm_base_url(self) -> str:
        """Base URL passed to the OpenAI-compatible chat model client."""

        if self.llm_provider == "ollama":
            return _normalize_ollama_base_url(self.ollama_base_url)
        return self.openai_base_url

    @property
    def effective_llm_model(self) -> str:
        """Model name passed to the chat model client."""

        if self.llm_provider == "ollama":
            return self.ollama_model
        return self.openai_model

    def require_llm_config(self) -> None:
        """Raise a clear error if the chat LLM is not configured."""

        if not self.has_llm_config:
            if self.llm_provider == "ollama":
                raise RuntimeError(
                    "Missing Ollama LLM configuration. Set OLLAMA_MODEL and "
                    "OLLAMA_BASE_URL in your environment or .env file before "
                    "running Agentic RAG with LLM_PROVIDER=ollama."
                )
            raise RuntimeError(
                "Missing LLM configuration. Set OPENAI_API_KEY and OPENAI_MODEL "
                "and OPENAI_BASE_URL in your environment or .env file before "
                "running Agentic RAG."
            )


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc


def _get_int_with_fallback(primary_name: str, fallback_name: str, default: int) -> int:
    raw_primary = os.getenv(primary_name)
    if raw_primary is not None and raw_primary.strip():
        return _get_int(primary_name, default)
    return _get_int(fallback_name, default)


def _get_float_with_fallback(
    primary_name: str,
    fallback_name: str,
    default: float,
) -> float:
    raw_primary = os.getenv(primary_name)
    if raw_primary is not None and raw_primary.strip():
        return _get_float(primary_name, default)
    return _get_float(fallback_name, default)


def _normalize_ollama_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def get_settings() -> Settings:
    """Load settings from environment variables."""

    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "openai_compatible").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "").strip(),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ).strip(),
        chunk_size=_get_int("CHUNK_SIZE", 800),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
        top_k=_get_int("TOP_K", 4),
        max_retry_count=_get_int_with_fallback(
            "MAX_RETRY_COUNT",
            "MAX_REWRITE_ATTEMPTS",
            2,
        ),
        temperature=_get_float_with_fallback(
            "LLM_TEMPERATURE",
            "OPENAI_TEMPERATURE",
            0,
        ),
        chroma_persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME",
            "agentic_rag_documents",
        ).strip(),
        gradio_server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1").strip(),
        gradio_server_port=_get_int("GRADIO_SERVER_PORT", 7860),
    )
