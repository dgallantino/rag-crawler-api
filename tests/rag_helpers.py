"""Shared helpers for RAG retrieval/generation tests."""

from __future__ import annotations

import math
import random
from datetime import datetime
from uuid import uuid4

from app.models import Document, DocumentChunk


def _vec(*values: float) -> list[float]:
    vector = [0.0] * 1536
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def _random_unit_vector(rng: random.Random) -> list[float]:
    vec = [rng.gauss(0.0, 1.0) for _ in range(1536)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


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


def bulk_make_chunks(
    db_session,
    collection,
    count: int,
    *,
    seed: int = 0,
    metadata: dict | None = None,
    metadata_every: int | None = None,
) -> list[DocumentChunk]:
    """Insert ``count`` synthetic chunks with random unit vectors.

    Creates a single parent document for efficiency. When ``metadata_every``
    is set (e.g. 2), every Nth chunk gets ``metadata``; others get {}.
    """
    rng = random.Random(seed)
    document = Document(
        collection_id=collection.id,
        url=f"file://bulk-{uuid4()}.md",
        title="bulk.md",
        content="bulk",
    )
    db_session.add(document)
    db_session.flush()

    created_at = datetime(2024, 6, 15, 12, 0, 0)
    chunks: list[DocumentChunk] = []
    for index in range(count):
        if metadata is not None and metadata_every is not None:
            chunk_metadata = metadata if index % metadata_every == 0 else {}
        else:
            chunk_metadata = metadata
        chunk = DocumentChunk(
            document_id=document.id,
            collection_id=collection.id,
            chunk_index=index,
            content=f"bulk-chunk-{index}",
            chunk_metadata=chunk_metadata,
            created_at=created_at,
            chunk_vector=_random_unit_vector(rng),
        )
        chunks.append(chunk)
        db_session.add(chunk)
        if (index + 1) % 500 == 0:
            db_session.flush()

    db_session.commit()
    return chunks
