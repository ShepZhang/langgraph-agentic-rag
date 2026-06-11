"""Document management routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from api.dependencies import get_document_service
from api.schemas import (
    DocumentDeleteResponse,
    DocumentIndexRequest,
    DocumentIndexResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from api.services.documents import DocumentService


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
def upload_documents(
    workspace_id: str = Form(default="default"),
    files: list[UploadFile] = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """Save uploaded documents and register them for indexing."""

    documents = document_service.upload_documents(workspace_id, files)
    return DocumentUploadResponse(documents=documents)


@router.post("/index", response_model=DocumentIndexResponse)
def index_documents(
    request: DocumentIndexRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentIndexResponse:
    """Index uploaded documents into the configured vector store."""

    return DocumentIndexResponse(
        **document_service.index_documents(
            workspace_id=request.workspace_id,
            document_ids=request.document_ids,
            reset_collection=request.reset_collection,
        )
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    workspace_id: str | None = None,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List API-managed documents."""

    return DocumentListResponse(
        documents=document_service.list_documents(workspace_id=workspace_id)
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentDeleteResponse:
    """Delete an API-managed document."""

    return DocumentDeleteResponse(
        **document_service.delete_document(document_id=document_id)
    )
