"""Hybrid retrieval over DocumentChunk (dense + Postgres FTS + RRF).

Responsible for turning a query into a ranked list of candidate chunks
from Postgres/pgvector and full-text search. Does not call any LLMs and
does not know about answer generation — see rerank.py and generation.py
for those. Optional Cohere rerank still happens in the service layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Callable
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import DocumentChunk


EmbedFn = Callable[[str], list[float]]


@dataclass
class RetrievedChunk:
    """A DocumentChunk plus its retrieval-stage score.

    For vector-only search the score is cosine similarity in ``[0, 1]``.
    For hybrid search it is the RRF fusion score (relative rank only).
    """

    chunk: DocumentChunk
    similarity_score: float


def embed_query(query: str, embed_fn: EmbedFn) -> list[float]:
    """Embed the query string.

    embed_fn is supplied by the service layer (the embedding model is
    out of scope here). This wrapper exists so retrieval.py has a
    single seam to mock in tests.
    """
    return embed_fn(query)


def _apply_hnsw_session_settings(session: Session, *, top_k: int) -> None:
    """Set LOCAL pgvector HNSW GUCs for the current transaction.

    ef_search scales with top_k so rerank over-fetch (top_k * 4 from the
    service layer) naturally requests a wider ANN candidate pool.

    PostgreSQL SET does not accept bind parameters, so values are inlined
    after validation.
    """
    settings = get_settings()
    ef_search = int(max(settings.hnsw_ef_search, top_k * 2))
    max_scan_tuples = int(settings.hnsw_max_scan_tuples)
    iterative_scan = settings.hnsw_iterative_scan
    if iterative_scan not in {"off", "strict_order", "relaxed_order"}:
        raise ValueError(f"invalid hnsw_iterative_scan: {iterative_scan!r}")

    session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))
    session.execute(text(f"SET LOCAL hnsw.iterative_scan = {iterative_scan}"))
    session.execute(text(f"SET LOCAL hnsw.max_scan_tuples = {max_scan_tuples}"))


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
    _apply_hnsw_session_settings(session, top_k=top_k)

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


def _build_tsquery(session: Session, query: str):
    """Prefer websearch_to_tsquery; fall back to plainto_tsquery if empty/invalid."""
    try:
        web = session.execute(
            select(func.websearch_to_tsquery("simple", query))
        ).scalar()
    except Exception:
        web = None

    if web is not None and str(web).strip():
        return func.websearch_to_tsquery("simple", query)

    return func.plainto_tsquery("simple", query)


def fts_search(
    session: Session,
    query: str,
    *,
    top_k: int,
    filters: dict | None,
    collection: list[str],
) -> list[RetrievedChunk]:
    """Run Postgres FTS over chunk content and return the top_k chunks.

    Uses the ``simple`` text search config (no English stemming) so Indonesian
    and exact tokens stay intact. Ranking uses ``ts_rank_cd``.
    """
    collection_ids = [UUID(c) for c in collection]
    tsquery = _build_tsquery(session, query)
    rank = func.ts_rank_cd(DocumentChunk.content_tsv, tsquery).label("rank")

    stmt = (
        select(DocumentChunk, rank)
        .where(DocumentChunk.collection_id.in_(collection_ids))
        .where(DocumentChunk.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(top_k)
    )

    if filters:
        stmt = _apply_filters(stmt, filters)

    rows = session.execute(stmt).all()
    return [
        RetrievedChunk(chunk=row[0], similarity_score=float(row[1] or 0.0))
        for row in rows
    ]


def rrf_fuse(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    k: int = 60,
    top_k: int,
) -> list[RetrievedChunk]:
    """Merge ranked retrieval lists with Reciprocal Rank Fusion.

    ``RRF(d) = sum_i 1 / (k + rank_i(d))`` with 1-based ranks. Chunks present
    in only one list still receive a score from that list. Ties break by
    chunk id (stable, ascending string form).
    """
    scores: dict[UUID, float] = {}
    chunks: dict[UUID, DocumentChunk] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            chunk_id = item.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk_id] = item.chunk

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return [
        RetrievedChunk(chunk=chunks[chunk_id], similarity_score=score)
        for chunk_id, score in ordered[:top_k]
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
    """Embed the query and return hybrid (or vector-only) candidates.

    When ``hybrid_search_enabled`` is true, runs dense + FTS legs in parallel
    (each fetching ``top_k * hybrid_candidate_multiplier``) and fuses with RRF.
    When disabled, preserves the previous vector-only path.
    """
    settings = get_settings()
    query_vector = embed_query(query, embed_fn)

    if not settings.hybrid_search_enabled:
        return vector_search(
            session,
            query_vector,
            top_k=top_k,
            filters=filters,
            collection=collection,
        )

    candidate_k = top_k * settings.hybrid_candidate_multiplier
    dense = vector_search(
        session,
        query_vector,
        top_k=candidate_k,
        filters=filters,
        collection=collection,
    )
    lexical = fts_search(
        session,
        query,
        top_k=candidate_k,
        filters=filters,
        collection=collection,
    )
    return rrf_fuse(
        [dense, lexical],
        k=settings.rrf_k,
        top_k=top_k,
    )
