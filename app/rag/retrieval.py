"""Vector similarity retrieval over DocumentChunk.

Responsible for turning a query embedding into a ranked list of
candidate chunks from Postgres/pgvector. Does not call any LLMs and
does not know about reranking or answer generation — see rerank.py and
generation.py for those.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import DocumentChunk


EmbedFn = Callable[[str], list[float]]


@dataclass
class RetrievedChunk:
    """A DocumentChunk plus its retrieval-stage similarity score."""

    chunk: DocumentChunk
    similarity_score: float  # similarity in [0, 1], 1 = identical


def embed_query(query: str, embed_fn: EmbedFn) -> list[float]:
    """Embed the query string.

    embed_fn is supplied by the service layer (the embedding model is
    out of scope here). This wrapper exists so retrieval.py has a
    single seam to mock in tests.
    """
    return embed_fn(query)


def vector_search(
    session: Session,
    query_vector: list[float],
    *,
    top_k: int,
    filters: dict | None,
    collection: list[str],
) -> list[RetrievedChunk]:
    """Run a pgvector similarity search and return the top_k chunks.

    ``collection`` is one or more collection UUIDs resolved by the service
    layer (tenant scoping happens there, not here).
    """
    collection_ids = [UUID(c) for c in collection]
    stmt = (
        select(
            DocumentChunk,
            DocumentChunk.chunk_vector.cosine_distance(query_vector).label("distance"),
        )
        .where(DocumentChunk.collection_id.in_(collection_ids))
        .order_by("distance")
        .limit(top_k)
    )

    if filters:
        stmt = _apply_filters(stmt, filters)

    rows = session.execute(stmt).all()
    return [
        RetrievedChunk(chunk=row[0], similarity_score=_distance_to_score(row[1]))
        for row in rows
    ]


def _apply_filters(stmt, filters: dict):
    """Translate `filters` into .where() clauses.

    `filters` follows the API layer's Filters model:
        {
            "metadata": dict[str, Any] | None,
            "date_range": {"after": date | None, "before": date | None} | None,
        }
    """
    metadata = filters.get("metadata")
    if metadata:
        for key, value in metadata.items():
            stmt = stmt.where(DocumentChunk.chunk_metadata.contains({key: value}))

    date_range = filters.get("date_range")
    if date_range:
        after = date_range.get("after")
        if after is not None:
            if isinstance(after, date) and not isinstance(after, datetime):
                after = datetime.combine(after, time.min)
            stmt = stmt.where(DocumentChunk.created_at >= after)
        before = date_range.get("before")
        if before is not None:
            if isinstance(before, date) and not isinstance(before, datetime):
                before = datetime.combine(before, time.max)
            stmt = stmt.where(DocumentChunk.created_at <= before)


    return stmt


def _distance_to_score(distance: float) -> float:
    """Convert cosine distance to a similarity score in [0, 1].

    cosine_distance() returns 1 - cosine_similarity, so similarity is 1 - distance.
    """
    return max(0.0, min(1.0, 1.0 - distance))


def retrieve(
    query: str,
    top_k: int,
    filters: dict | None,
    collection: list[str],
    *,
    session: Session,
    embed_fn: EmbedFn,
) -> list[RetrievedChunk]:
    """Embed the query and return similarity candidates for ``collection``."""

    query_vector = embed_query(query, embed_fn)

    candidates = vector_search(
        session,
        query_vector,
        top_k=top_k,
        filters=filters,
        collection=collection,
    )

    return candidates
