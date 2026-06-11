"""Document upload registry and indexing service."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from config import Settings, get_settings
from rag.chunker import split_documents
from rag.loader import load_documents
from rag.vectorstore import build_document_ids, get_vectorstore_manager


class DocumentService:
    """Manage API-uploaded documents."""

    def __init__(
        self,
        settings: Settings | None = None,
        vectorstore_manager: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vectorstore_manager = vectorstore_manager or get_vectorstore_manager()

    def upload_documents(
        self,
        workspace_id: str,
        files: list[UploadFile],
    ) -> list[dict[str, Any]]:
        """Save uploaded files and create registry records."""

        if not files:
            raise HTTPException(status_code=400, detail="Upload at least one file.")

        registry = self._read_registry()
        records: list[dict[str, Any]] = []
        for uploaded_file in files:
            filename = _safe_filename(uploaded_file.filename)
            document_id = f"doc_{uuid.uuid4().hex}"
            directory = self.settings.api_upload_dir / workspace_id / document_id
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / filename
            path.write_bytes(uploaded_file.file.read())
            record = {
                "document_id": document_id,
                "workspace_id": workspace_id,
                "filename": filename,
                "path": str(path),
                "status": "uploaded",
                "chunk_count": 0,
                "vector_ids": [],
            }
            registry[document_id] = record
            records.append(_public_document(record))

        self._write_registry(registry)
        return records

    def index_documents(
        self,
        workspace_id: str,
        document_ids: list[str] | None = None,
        reset_collection: bool = False,
    ) -> dict[str, Any]:
        """Index uploaded documents into the configured vector store."""

        registry = self._read_registry()
        selected_records = self._select_records(registry, workspace_id, document_ids)
        if not selected_records:
            raise HTTPException(status_code=404, detail="No matching documents found.")

        documents = load_documents(record["path"] for record in selected_records)
        for document in documents:
            source_path = str(document.metadata.get("source_path", ""))
            owner = _find_record_by_path(selected_records, source_path)
            if owner:
                document.metadata["document_id"] = owner["document_id"]
                document.metadata["workspace_id"] = owner["workspace_id"]

        chunks = split_documents(documents)
        if reset_collection:
            self.vectorstore_manager.create_vectorstore(
                chunks,
                reset_collection=True,
            )
        else:
            self.vectorstore_manager.add_documents(chunks)

        vector_ids = build_document_ids(chunks)
        by_document_id: dict[str, list[str]] = {}
        for chunk, vector_id in zip(chunks, vector_ids, strict=True):
            document_id = str(chunk.metadata.get("document_id", ""))
            by_document_id.setdefault(document_id, []).append(vector_id)

        indexed_records: list[dict[str, Any]] = []
        for record in selected_records:
            record["status"] = "indexed"
            record["vector_ids"] = by_document_id.get(record["document_id"], [])
            record["chunk_count"] = len(record["vector_ids"])
            registry[record["document_id"]] = record
            indexed_records.append(_public_document(record))

        self._write_registry(registry)
        return {
            "workspace_id": workspace_id,
            "indexed_documents": indexed_records,
            "chunk_count": sum(record["chunk_count"] for record in indexed_records),
            "reset_collection": reset_collection,
        }

    def list_documents(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        """List API-managed documents."""

        records = self._read_registry().values()
        if workspace_id:
            records = [
                record
                for record in records
                if record.get("workspace_id") == workspace_id
            ]
        return [_public_document(record) for record in records]

    def delete_document(self, document_id: str) -> dict[str, Any]:
        """Delete a registered document and known vector IDs."""

        registry = self._read_registry()
        record = registry.pop(document_id, None)
        if record is None:
            raise HTTPException(status_code=404, detail="Document not found.")

        vector_ids = list(record.get("vector_ids", []))
        deleted_vector_count = 0
        if vector_ids and hasattr(self.vectorstore_manager, "delete_documents"):
            deleted_vector_count = self.vectorstore_manager.delete_documents(vector_ids)

        path = Path(record.get("path", ""))
        if path.exists():
            path.unlink()
            shutil.rmtree(path.parent, ignore_errors=True)

        self._write_registry(registry)
        return {
            "document_id": document_id,
            "deleted": True,
            "deleted_vector_count": deleted_vector_count,
        }

    def _select_records(
        self,
        registry: dict[str, dict[str, Any]],
        workspace_id: str,
        document_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        records = [
            record
            for record in registry.values()
            if record.get("workspace_id") == workspace_id
        ]
        if document_ids is None:
            return records
        selected = set(document_ids)
        return [record for record in records if record.get("document_id") in selected]

    def _read_registry(self) -> dict[str, dict[str, Any]]:
        path = self.settings.api_document_registry_path
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): value
            for key, value in payload.items()
            if isinstance(value, dict)
        }

    def _write_registry(self, registry: dict[str, dict[str, Any]]) -> None:
        path = self.settings.api_document_registry_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _safe_filename(filename: str | None) -> str:
    safe_name = Path(filename or "uploaded_document").name
    return safe_name or "uploaded_document"


def _public_document(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": record["document_id"],
        "workspace_id": record["workspace_id"],
        "filename": record["filename"],
        "status": record["status"],
        "chunk_count": int(record.get("chunk_count", 0) or 0),
        "vector_ids": list(record.get("vector_ids", []) or []),
    }


def _find_record_by_path(
    records: list[dict[str, Any]],
    source_path: str,
) -> dict[str, Any] | None:
    for record in records:
        if str(Path(record["path"]).resolve()) == source_path:
            return record
    return None
