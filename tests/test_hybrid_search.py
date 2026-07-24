"""Tests for hybrid retrieval (Postgres FTS + dense + RRF).

Runs against Postgres/pgvector via the existing Podman-backed testcontainers
setup in conftest (do not start pods ad-hoc for these tests).
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.config import get_settings
from app.models import DocumentChunk
from app.rag.retrieval import (
    RetrievedChunk,
    fts_search,
    retrieve,
    rrf_fuse,
)
from tests.rag_helpers import _make_chunk, _vec


GIN_INDEX_NAME = "ix_document_chunks_content_tsv_gin"


def test_content_tsv_gin_index_exists_after_create_all(db_session) -> None:
    row = db_session.execute(
        text(
            """
            SELECT i.relname AS index_name,
                   am.amname AS access_method
            FROM pg_class t
            JOIN pg_index ix ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_am am ON i.relam = am.oid
            WHERE t.relname = 'document_chunks'
              AND i.relname = :index_name
            """
        ),
        {"index_name": GIN_INDEX_NAME},
    ).one_or_none()

    assert row is not None, f"expected index {GIN_INDEX_NAME}"
    assert row.access_method == "gin"


def test_content_tsv_generated_from_content(db_session, test_collection) -> None:
    chunk = _make_chunk(
        db_session,
        test_collection,
        content="rare-token-xyz hospital poliklinik",
        chunk_vector=_vec(1.0),
    )
    db_session.refresh(chunk)

    assert chunk.content_tsv is not None

    expected = db_session.execute(
        text("SELECT to_tsvector('simple', :content)"),
        {"content": chunk.content},
    ).scalar_one()
    actual = db_session.execute(
        text("SELECT content_tsv FROM document_chunks WHERE id = :id"),
        {"id": chunk.id},
    ).scalar_one()
    assert actual == expected


def test_fts_search_ranks_exact_token_above_unrelated(
    db_session, test_collection
) -> None:
    match = _make_chunk(
        db_session,
        test_collection,
        content="Invoice ZX9Q7-TOKEN for radiology",
        chunk_vector=_vec(0.0, 1.0),
        chunk_index=0,
    )
    _make_chunk(
        db_session,
        test_collection,
        content="Unrelated gardening tips and weather notes",
        chunk_vector=_vec(1.0),
        chunk_index=1,
    )

    results = fts_search(
        db_session,
        "ZX9Q7-TOKEN",
        top_k=5,
        filters=None,
        collection=[str(test_collection.id)],
    )

    assert len(results) >= 1
    assert results[0].chunk.id == match.id


def test_fts_search_scopes_by_collection(
    db_session, test_user, test_collection
) -> None:
    user, _ = test_user
    from app.services.collections import create_collection

    other = create_collection(
        db_session, user, name="Other", slug="other-fts"
    )
    in_scope = _make_chunk(
        db_session,
        test_collection,
        content="shared-rare-token alpha",
        chunk_vector=_vec(1.0),
    )
    _make_chunk(
        db_session,
        other,
        content="shared-rare-token beta",
        chunk_vector=_vec(1.0),
    )

    results = fts_search(
        db_session,
        "shared-rare-token",
        top_k=5,
        filters=None,
        collection=[str(test_collection.id)],
    )

    assert len(results) == 1
    assert results[0].chunk.id == in_scope.id


def test_fts_search_applies_metadata_and_date_filters(
    db_session, test_collection
) -> None:
    match = _make_chunk(
        db_session,
        test_collection,
        content="policy ZXFTS1 document",
        metadata={"doc_type": "contract"},
        created_at=datetime(2024, 6, 15, 12, 0, 0),
        chunk_vector=_vec(1.0),
        chunk_index=0,
    )
    _make_chunk(
        db_session,
        test_collection,
        content="policy ZXFTS1 memo",
        metadata={"doc_type": "memo"},
        created_at=datetime(2024, 6, 15, 12, 0, 0),
        chunk_vector=_vec(1.0),
        chunk_index=1,
    )
    _make_chunk(
        db_session,
        test_collection,
        content="policy ZXFTS1 old",
        metadata={"doc_type": "contract"},
        created_at=datetime(2023, 1, 1, 0, 0, 0),
        chunk_vector=_vec(1.0),
        chunk_index=2,
    )

    results = fts_search(
        db_session,
        "ZXFTS1",
        top_k=5,
        filters={
            "metadata": {"doc_type": "contract"},
            "date_range": {"after": date(2024, 6, 1)},
        },
        collection=[str(test_collection.id)],
    )

    assert len(results) == 1
    assert results[0].chunk.id == match.id


def test_rrf_fuse_prefers_dual_high_ranks_and_keeps_singletons() -> None:
    a = DocumentChunk(id=uuid4(), content="a", chunk_index=0, chunk_vector=[0.0])
    b = DocumentChunk(id=uuid4(), content="b", chunk_index=1, chunk_vector=[0.0])
    c = DocumentChunk(id=uuid4(), content="c", chunk_index=2, chunk_vector=[0.0])

    dense = [
        RetrievedChunk(chunk=a, similarity_score=0.9),
        RetrievedChunk(chunk=b, similarity_score=0.8),
    ]
    lexical = [
        RetrievedChunk(chunk=a, similarity_score=0.7),
        RetrievedChunk(chunk=c, similarity_score=0.6),
    ]

    fused = rrf_fuse([dense, lexical], k=60, top_k=3)

    assert fused[0].chunk.id == a.id
    assert {fused[1].chunk.id, fused[2].chunk.id} == {b.id, c.id}
    assert fused[0].similarity_score > fused[1].similarity_score
    assert fused[1].similarity_score == pytest.approx(fused[2].similarity_score)


def test_rrf_fuse_stable_tie_break_by_chunk_id() -> None:
    # Same rank in separate singleton lists → equal RRF; order by id string.
    lower_id = uuid4()
    higher_id = uuid4()
    first_id, second_id = sorted([lower_id, higher_id], key=str)

    chunk_first = DocumentChunk(
        id=first_id, content="x", chunk_index=0, chunk_vector=[0.0]
    )
    chunk_second = DocumentChunk(
        id=second_id, content="y", chunk_index=1, chunk_vector=[0.0]
    )

    fused = rrf_fuse(
        [
            [RetrievedChunk(chunk=chunk_first, similarity_score=1.0)],
            [RetrievedChunk(chunk=chunk_second, similarity_score=1.0)],
        ],
        k=60,
        top_k=2,
    )

    assert [item.chunk.id for item in fused] == [first_id, second_id]


def test_retrieve_hybrid_fuses_vector_and_fts(
    db_session, test_collection, monkeypatch
) -> None:
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "hybrid_search_enabled", True)
    monkeypatch.setattr(settings, "hybrid_candidate_multiplier", 2)
    monkeypatch.setattr(settings, "rrf_k", 60)
    monkeypatch.setattr("app.rag.retrieval.get_settings", lambda: settings)

    # Dense-favored: close to query embedding, no rare token.
    vector_favored = _make_chunk(
        db_session,
        test_collection,
        content="semantic paraphrase about billing process",
        chunk_vector=_vec(1.0),
        chunk_index=0,
    )
    # FTS-favored: exact rare token, orthogonal embedding.
    fts_favored = _make_chunk(
        db_session,
        test_collection,
        content="Contains UNIQUELEXEME99 only here",
        chunk_vector=_vec(0.0, 1.0),
        chunk_index=1,
    )

    results = retrieve(
        query="UNIQUELEXEME99",
        top_k=2,
        filters=None,
        collection=[str(test_collection.id)],
        session=db_session,
        embed_fn=lambda _q: _vec(1.0),
    )

    ids = [r.chunk.id for r in results]
    assert fts_favored.id in ids
    assert vector_favored.id in ids
    # Exact token should rank at least as high as the semantic-only chunk.
    assert ids.index(fts_favored.id) <= ids.index(vector_favored.id)

    get_settings.cache_clear()


def test_retrieve_hybrid_disabled_is_vector_only(
    db_session, test_collection, monkeypatch
) -> None:
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "hybrid_search_enabled", False)
    monkeypatch.setattr("app.rag.retrieval.get_settings", lambda: settings)

    closer = _make_chunk(
        db_session,
        test_collection,
        content="no shared tokens with the query string",
        chunk_vector=_vec(1.0),
        chunk_index=0,
    )
    _make_chunk(
        db_session,
        test_collection,
        content="UNIQUELEXEME99 appears here for FTS",
        chunk_vector=_vec(0.0, 1.0),
        chunk_index=1,
    )

    results = retrieve(
        query="UNIQUELEXEME99",
        top_k=1,
        filters=None,
        collection=[str(test_collection.id)],
        session=db_session,
        embed_fn=lambda _q: _vec(1.0),
    )

    assert len(results) == 1
    assert results[0].chunk.id == closer.id
    assert results[0].similarity_score == pytest.approx(1.0, abs=1e-4)

    get_settings.cache_clear()
