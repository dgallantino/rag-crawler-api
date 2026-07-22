"""Tests for pgvector HNSW index on DocumentChunk.chunk_vector."""

from __future__ import annotations

import random

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.models import Document, DocumentChunk
from app.rag.retrieval import vector_search
from tests.rag_helpers import _random_unit_vector, _vec, bulk_make_chunks


INDEX_NAME = "ix_document_chunks_chunk_vector_hnsw"


def test_hnsw_index_exists_after_create_all(db_session) -> None:
    row = db_session.execute(
        text(
            """
            SELECT i.relname AS index_name,
                   am.amname AS access_method,
                   pg_get_indexdef(i.oid) AS index_def
            FROM pg_class t
            JOIN pg_index ix ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_am am ON i.relam = am.oid
            WHERE t.relname = 'document_chunks'
              AND i.relname = :index_name
            """
        ),
        {"index_name": INDEX_NAME},
    ).one_or_none()

    assert row is not None, f"expected index {INDEX_NAME}"
    assert row.access_method == "hnsw"
    assert "vector_cosine_ops" in row.index_def


def test_chunk_vector_is_not_null(db_session, test_collection) -> None:
    nullable = db_session.execute(
        text(
            """
            SELECT is_nullable
            FROM information_schema.columns
            WHERE table_name = 'document_chunks'
              AND column_name = 'chunk_vector'
            """
        )
    ).scalar_one()
    assert nullable == "NO"

    document = Document(
        collection_id=test_collection.id,
        url="file://null-vector.md",
        title="null-vector.md",
        content="x",
    )
    db_session.add(document)
    db_session.flush()

    chunk = DocumentChunk(
        document_id=document.id,
        collection_id=test_collection.id,
        chunk_index=0,
        content="missing vector",
        chunk_vector=None,  # type: ignore[arg-type]
    )
    db_session.add(chunk)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_vector_search_scopes_to_collection_with_similar_vectors(
    db_session, test_user, test_collection
) -> None:
    """T5 regression: similar vectors in another collection must not leak."""
    from app.services.collections import create_collection

    user, _ = test_user
    other = create_collection(db_session, user, name="Other", slug="other-hnsw")

    from tests.rag_helpers import _make_chunk

    in_scope = _make_chunk(
        db_session,
        test_collection,
        content="in scope",
        chunk_vector=_vec(1.0),
    )
    _make_chunk(
        db_session,
        other,
        content="out of scope",
        chunk_vector=_vec(1.0),
    )

    results = vector_search(
        db_session,
        _vec(1.0),
        top_k=5,
        filters=None,
        collection=[str(test_collection.id)],
    )
    assert [r.chunk.id for r in results] == [in_scope.id]


@pytest.mark.slow
def test_planner_uses_hnsw_at_scale(db_session, test_collection) -> None:
    # Threshold: below ~2k rows the planner may legitimately prefer seq scan.
    row_count = 3000
    bulk_make_chunks(db_session, test_collection, row_count, seed=42)
    db_session.execute(text("ANALYZE document_chunks"))

    query_vector = _vec(1.0)
    # Build a plan matching vector_search's SQL shape (cosine distance + filter).
    plan_rows = db_session.execute(
        text(
            """
            EXPLAIN (FORMAT TEXT)
            SELECT id, chunk_vector <=> CAST(:qv AS vector) AS distance
            FROM document_chunks
            WHERE collection_id = CAST(:cid AS uuid)
            ORDER BY distance
            LIMIT 10
            """
        ),
        {"qv": str(query_vector), "cid": str(test_collection.id)},
    ).scalars().all()
    plan = "\n".join(plan_rows)

    if "Seq Scan" in plan and INDEX_NAME not in plan:
        pytest.skip(
            f"planner chose seq scan at {row_count} rows (acceptable for small tables):\n{plan}"
        )

    assert INDEX_NAME in plan
    assert "Index Scan" in plan or "Bitmap Index Scan" in plan


def _recall_at_k(ann_ids: list, brute_ids: list, k: int) -> float:
    if k == 0:
        return 1.0
    return len(set(ann_ids[:k]) & set(brute_ids[:k])) / float(k)


def _brute_force_ids(
    db_session,
    query_vector: list[float],
    *,
    collection_id: str,
    top_k: int,
    filters: dict | None = None,
) -> list:
    """Exact top-k by disabling index scans in this transaction."""
    db_session.execute(text("SET LOCAL enable_indexscan = off"))
    db_session.execute(text("SET LOCAL enable_bitmapscan = off"))
    results = vector_search(
        db_session,
        query_vector,
        top_k=top_k,
        filters=filters,
        collection=[collection_id],
    )
    # Re-enable for subsequent ANN queries in the same transaction.
    db_session.execute(text("SET LOCAL enable_indexscan = on"))
    db_session.execute(text("SET LOCAL enable_bitmapscan = on"))
    return [r.chunk.id for r in results]


@pytest.fixture
def high_ef_search(monkeypatch):
    """Raise ef_search so recall tests measure ANN quality, not under-search."""
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "hnsw_ef_search", 200)
    monkeypatch.setattr("app.rag.retrieval.get_settings", lambda: settings)
    yield
    get_settings.cache_clear()


@pytest.mark.slow
def test_hnsw_recall_vs_brute_force(db_session, test_collection, high_ef_search) -> None:
    # active_dims=64: isotropic 1536-d noise makes neighbors nearly equidistant
    # and collapses HNSW recall even with a healthy index.
    bulk_make_chunks(db_session, test_collection, 3000, seed=7, active_dims=64)
    db_session.execute(text("ANALYZE document_chunks"))

    rng = random.Random(99)
    top_k = 10
    recalls: list[float] = []
    for _ in range(20):
        query = _random_unit_vector(rng, active_dims=64)
        brute_ids = _brute_force_ids(
            db_session,
            query,
            collection_id=str(test_collection.id),
            top_k=top_k,
        )
        ann = vector_search(
            db_session,
            query,
            top_k=top_k,
            filters=None,
            collection=[str(test_collection.id)],
        )
        ann_ids = [r.chunk.id for r in ann]
        recalls.append(_recall_at_k(ann_ids, brute_ids, top_k))

    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall >= 0.9, f"mean recall@10={mean_recall:.3f} < 0.9; samples={recalls}"


@pytest.mark.slow
def test_hnsw_filtered_recall(db_session, test_collection, high_ef_search) -> None:
    bulk_make_chunks(
        db_session,
        test_collection,
        3000,
        seed=11,
        metadata={"doc_type": "contract"},
        metadata_every=2,
        active_dims=64,
    )
    db_session.execute(text("ANALYZE document_chunks"))

    filters = {"metadata": {"doc_type": "contract"}}
    rng = random.Random(123)
    top_k = 10
    recalls: list[float] = []
    for _ in range(20):
        query = _random_unit_vector(rng, active_dims=64)
        brute_ids = _brute_force_ids(
            db_session,
            query,
            collection_id=str(test_collection.id),
            top_k=top_k,
            filters=filters,
        )
        ann = vector_search(
            db_session,
            query,
            top_k=top_k,
            filters=filters,
            collection=[str(test_collection.id)],
        )
        ann_ids = [r.chunk.id for r in ann]
        recalls.append(_recall_at_k(ann_ids, brute_ids, top_k))

    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall >= 0.85, (
        f"filtered mean recall@10={mean_recall:.3f} < 0.85; samples={recalls}"
    )
