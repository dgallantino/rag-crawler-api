"""Shared helpers for RAG retrieval/generation tests."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from app.models import Document, DocumentChunk


def _vec(*values: float) -> list[float]:
    vector = [0.0] * 1536
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def _make_chunk(
    db_session,
    collection,
    *,
    content: str = "chunk content",
    metadata: dict | None = None,
    created_at: datetime | None = None,
    chunk_vector: list[float] | None = None,
    chunk_index: int = 0,
) -> DocumentChunk:
    document = Document(
        collection_id=collection.id,
        url=f"file://{uuid4()}.md",
        title="test.md",
        content=content,
    )
    db_session.add(document)
    db_session.flush()

    chunk = DocumentChunk(
        document_id=document.id,
        collection_id=collection.id,
        chunk_index=chunk_index,
        content=content,
        chunk_metadata=metadata,
        created_at=created_at or datetime(2024, 6, 15, 12, 0, 0),
        chunk_vector=chunk_vector or ([0.1] * 1536),
    )
    db_session.add(chunk)
    db_session.commit()
    return chunk
