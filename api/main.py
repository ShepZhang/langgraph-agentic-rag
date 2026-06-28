"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from api.routes import chat, documents, evaluation


def create_app() -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(
        title="Reliability-oriented Agentic RAG API",
        version="0.5.1-p5b",
    )
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(evaluation.router)
    return app


app = create_app()
