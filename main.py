"""Command-line helper for the Agentic RAG document QA system."""

from __future__ import annotations

from config import get_settings


def main() -> None:
    """Print a concise configuration summary."""

    settings = get_settings()
    print("Agentic RAG Document QA System")
    print(f"LLM model: {settings.openai_model}")
    print(f"LLM configured: {settings.has_llm_config}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Chroma path: {settings.chroma_persist_dir}")
    print(f"Top K: {settings.top_k}")
    print(f"Max rewrite attempts: {settings.max_rewrite_attempts}")
    print("Run `.venv/bin/python app.py` to start the Gradio UI.")


if __name__ == "__main__":
    main()
