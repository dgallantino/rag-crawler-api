"""Hybrid retrieval over DocumentChunk (dense + Postgres FTS + RRF).

Responsible for turning a query into a ranked list of candidate chunks
from Postgres/pgvector and full-text search, with optional reranking
orchestrated by ``Retriever``. Does not call any LLMs and does not know
about answer generation — see generation.py for that. The embedding and
rerank providers are injected by the service layer.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from typing import Callable
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import DocumentChunk
from app.rag.chunks import ScoredChunk

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], list[float]]

# Contract for the service-layer rerank provider:
# (query, top_k, candidates) -> reranked chunks with rerank_score set.
RerankServiceFn = Callable[[str, int, list[ScoredChunk]], list[ScoredChunk]]


def _embed_query(query: str, embed_fn: EmbedFn) -> list[float]:
    """Embed the query string.

    embed_fn is supplied by the service layer (the embedding model is
    out of scope here). This wrapper exists so retrieval.py has a
    single seam to mock in tests.
    """
    return embed_fn(query)


def _apply_hnsw_session_settings(session: Session, *, top_k: int) -> None:
    """Set LOCAL pgvector HNSW GUCs for the current transaction.

    ef_search scales with top_k so rerank over-fetch (top_k *
    rerank_expansion_factor in Retriever) naturally requests a wider ANN
    candidate pool.

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


def _vector_search(
    session: Session,
    query_vector: list[float],
    *,
    top_k: int,
    filters: dict | None,
    collection: list[str],
) -> list[ScoredChunk]:
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
        ScoredChunk(
            chunk=row[0],
            score=_distance_to_score(row[1]),
            retrieval_score=_distance_to_score(row[1]),
        )
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


def _fts_search(
    session: Session,
    query: str,
    *,
    top_k: int,
    filters: dict | None,
    collection: list[str],
) -> list[ScoredChunk]:
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
        ScoredChunk(
            chunk=row[0],
            score=float(row[1] or 0.0),
            retrieval_score=float(row[1] or 0.0),
        )
        for row in rows
    ]


def _rrf_fuse(
    ranked_lists: list[list[ScoredChunk]],
    *,
    k: int = 60,
    top_k: int,
) -> list[ScoredChunk]:
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
        ScoredChunk(chunk=chunks[chunk_id], score=score, retrieval_score=score)
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


def _retrieve(
    query: str,
    top_k: int,
    filters: dict | None,
    collection: list[str],
    *,
    session: Session,
    embed_fn: EmbedFn,
) -> list[ScoredChunk]:
    """Embed the query and return hybrid (or vector-only) candidates.

    When ``hybrid_search_enabled`` is true, runs dense + FTS legs
    (each fetching ``top_k * hybrid_candidate_multiplier``) and fuses with RRF.
    When disabled, preserves the vector-only path.
    """
    settings = get_settings()
    query_vector = _embed_query(query, embed_fn)

    if not settings.hybrid_search_enabled:
        return _vector_search(
            session,
            query_vector,
            top_k=top_k,
            filters=filters,
            collection=collection,
        )

    candidate_k = top_k * settings.hybrid_candidate_multiplier
    dense = _vector_search(
        session,
        query_vector,
        top_k=candidate_k,
        filters=filters,
        collection=collection,
    )
    lexical = _fts_search(
        session,
        query,
        top_k=candidate_k,
        filters=filters,
        collection=collection,
    )
    return _rrf_fuse(
        [dense, lexical],
        k=settings.rrf_k,
        top_k=top_k,
    )


class Retriever:
    """Orchestrates hybrid retrieval and optional reranking.

    Holds the injected embedding/rerank providers plus the rerank over-fetch
    policy, so it can be constructed once (per process) by the service layer.
    The DB session is per-request state and is always passed per call.
    """

    def __init__(
        self,
        *,
        embed_fn: EmbedFn,
        rerank_fn: RerankServiceFn | None = None,
        rerank_expansion_factor: int = 4,
    ) -> None:
        self._embed_fn = embed_fn
        self._rerank_fn = rerank_fn
        self._rerank_expansion_factor = rerank_expansion_factor

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict | None,
        collection: list[str],
        *,
        session: Session,
        use_rerank: bool = False,
    ) -> list[ScoredChunk]:
        """Retrieve top_k chunks, over-fetching then reranking when requested."""
        initial_k = top_k * self._rerank_expansion_factor if use_rerank else top_k
        candidates = _retrieve(
            query,
            initial_k,
            filters,
            collection,
            session=session,
            embed_fn=self._embed_fn,
        )

        if not use_rerank:
            return candidates[:top_k]
        return self._rerank(query, candidates, top_k)

    def _rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Re-score candidates; on any failure fall back to retrieval order."""
        if self._rerank_fn is None:
            logger.warning(
                "use_rerank requested but no rerank provider configured; "
                "falling back to retrieval order"
            )
            return candidates[:top_k]

        try:
            reranked = self._rerank_fn(query, top_k, candidates)
        except Exception:
            logger.warning(
                "Rerank failed; falling back to retrieval order", exc_info=True
            )
            return candidates[:top_k]

        return reranked[:top_k]
