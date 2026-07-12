"""Tests for RAG retrieval and generation modules.

Retrieval and filter tests run against Postgres/pgvector via testcontainers.
Generation unit tests use mocked completion clients — no live API calls.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.models import Document, DocumentChunk
from app.rag.generation import (
    ScoredChunk,
    answer_with_retrieval,
    build_context,
    generate_answer,
    normalize_chunks,
)
from app.rag.retrieval import (
    RetrievedChunk,
    _apply_filters,
    _distance_to_score,
    embed_query,
    retrieve,
    vector_search,
)


def fake_embed_fn(_query: str) -> list[float]:
    return [0.0] * 1536


def fake_completion_client(answer: str = "Mock answer") -> MagicMock:
    mock = MagicMock()

    def create(*, model, messages, **kwargs):
        return MagicMock(choices=[MagicMock(message=MagicMock(content=answer))])

    mock.chat.completions.create = create
    return mock


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


def _retrieved(chunk: DocumentChunk, similarity_score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, similarity_score=similarity_score)


def _scored(chunk: DocumentChunk, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score)


# --- retrieval.py: _distance_to_score ---


def test_distance_to_score_clamps():
    assert _distance_to_score(0.0) == 1.0
    assert _distance_to_score(1.0) == 0.0
    assert _distance_to_score(1.5) == 0.0
    assert _distance_to_score(-0.2) == 1.0


# --- retrieval.py: _apply_filters ---


def test_apply_filters_metadata_scalar(db_session, test_collection):
    match = _make_chunk(
        db_session,
        test_collection,
        content="contract chunk",
        metadata={"doc_type": "contract"},
    )
    _make_chunk(
        db_session,
        test_collection,
        content="other chunk",
        metadata={"doc_type": "memo"},
    )

    stmt = select(DocumentChunk)
    stmt = _apply_filters(stmt, {"metadata": {"doc_type": "contract"}})
    rows = db_session.execute(stmt).scalars().all()

    assert len(rows) == 1
    assert rows[0].id == match.id


def test_apply_filters_metadata_applies_one_clause_per_key(db_session, test_collection):
    """Each metadata key becomes its own .contains() where clause."""
    _make_chunk(
        db_session,
        test_collection,
        metadata={"doc_type": "contract", "region": "eu"},
    )

    stmt = select(DocumentChunk)
    stmt = _apply_filters(
        stmt,
        {"metadata": {"doc_type": "contract", "region": "eu"}},
    )
    where_sql = " ".join(str(clause) for clause in stmt._where_criteria)

    assert "metadata" in where_sql
    assert len(stmt._where_criteria) == 2


def test_apply_filters_date_range(db_session, test_collection):
    early = _make_chunk(
        db_session,
        test_collection,
        content="early",
        created_at=datetime(2024, 1, 10, 0, 0, 0),
    )
    mid = _make_chunk(
        db_session,
        test_collection,
        content="mid",
        created_at=datetime(2024, 6, 15, 12, 0, 0),
    )
    _make_chunk(
        db_session,
        test_collection,
        content="late",
        created_at=datetime(2024, 12, 10, 0, 0, 0),
    )

    stmt = select(DocumentChunk)
    stmt = _apply_filters(
        stmt,
        {"date_range": {"after": date(2024, 6, 1), "before": date(2024, 6, 30)}},
    )
    rows = db_session.execute(stmt).scalars().all()
    assert len(rows) == 1
    assert rows[0].id == mid.id

    stmt = select(DocumentChunk)
    stmt = _apply_filters(stmt, {"date_range": {"after": date(2024, 1, 1)}})
    rows = db_session.execute(stmt).scalars().all()
    assert len(rows) == 3

    stmt = select(DocumentChunk)
    stmt = _apply_filters(stmt, {"date_range": {"before": date(2024, 2, 1)}})
    rows = db_session.execute(stmt).scalars().all()
    assert len(rows) == 1
    assert rows[0].id == early.id


def test_apply_filters_date_range_accepts_datetime(db_session, test_collection):
    boundary = _make_chunk(
        db_session,
        test_collection,
        content="boundary",
        created_at=datetime(2024, 6, 1, 15, 30, 0),
    )

    stmt = select(DocumentChunk)
    stmt = _apply_filters(
        stmt,
        {
            "date_range": {
                "after": datetime(2024, 6, 1, 0, 0, 0),
                "before": datetime(2024, 6, 1, 23, 59, 59),
            }
        },
    )
    rows = db_session.execute(stmt).scalars().all()

    assert len(rows) == 1
    assert rows[0].id == boundary.id


def test_apply_filters_date_only_expands_to_day_bounds(db_session, test_collection):
    """Plain date values should cover the full calendar day."""
    on_day = _make_chunk(
        db_session,
        test_collection,
        content="on day",
        created_at=datetime(2024, 6, 15, 23, 59, 59),
    )

    stmt = select(DocumentChunk)
    stmt = _apply_filters(
        stmt,
        {"date_range": {"after": date(2024, 6, 15), "before": date(2024, 6, 15)}},
    )
    rows = db_session.execute(stmt).scalars().all()

    assert len(rows) == 1
    assert rows[0].id == on_day.id


def test_apply_filters_owner_ref_skipped(db_session, test_collection):
    chunk = _make_chunk(db_session, test_collection, content="owned")
    stmt = select(DocumentChunk)
    stmt = _apply_filters(stmt, {"owner_ref": "user-123"})
    rows = db_session.execute(stmt).scalars().all()
    assert len(rows) == 1
    assert rows[0].id == chunk.id


def test_apply_filters_combined_metadata_and_date(db_session, test_collection):
    match = _make_chunk(
        db_session,
        test_collection,
        content="match",
        metadata={"doc_type": "contract"},
        created_at=datetime(2024, 6, 10, 0, 0, 0),
    )
    _make_chunk(
        db_session,
        test_collection,
        content="wrong type",
        metadata={"doc_type": "memo"},
        created_at=datetime(2024, 6, 10, 0, 0, 0),
    )
    _make_chunk(
        db_session,
        test_collection,
        content="wrong date",
        metadata={"doc_type": "contract"},
        created_at=datetime(2024, 1, 1, 0, 0, 0),
    )

    stmt = select(DocumentChunk)
    stmt = _apply_filters(
        stmt,
        {
            "metadata": {"doc_type": "contract"},
            "date_range": {"after": date(2024, 6, 1)},
        },
    )
    rows = db_session.execute(stmt).scalars().all()

    assert len(rows) == 1
    assert rows[0].id == match.id


def test_apply_filters_empty_dict_is_noop(db_session, test_collection):
    chunk = _make_chunk(db_session, test_collection, content="any")
    stmt = select(DocumentChunk)
    stmt = _apply_filters(stmt, {})
    rows = db_session.execute(stmt).scalars().all()
    assert len(rows) == 1
    assert rows[0].id == chunk.id


# --- retrieval.py: vector_search ---


def test_vector_search_maps_distance_to_similarity_score(db_session, test_collection):
    closer = _make_chunk(
        db_session,
        test_collection,
        content="Alpha",
        chunk_vector=_vec(1.0),
    )
    _make_chunk(
        db_session,
        test_collection,
        content="Beta",
        chunk_vector=_vec(0.0, 1.0),
    )

    results = vector_search(
        db_session,
        _vec(1.0),
        top_k=2,
        filters=None,
        collection=None,
    )

    assert len(results) == 2
    assert results[0].chunk.id == closer.id
    assert results[0].similarity_score == pytest.approx(1.0, abs=1e-5)
    assert results[1].similarity_score < results[0].similarity_score


def test_vector_search_respects_top_k(db_session, test_collection):
    for index in range(3):
        _make_chunk(
            db_session,
            test_collection,
            content=f"chunk-{index}",
            chunk_index=index,
            chunk_vector=_vec(1.0 - index * 0.1),
        )

    results = vector_search(
        db_session,
        _vec(1.0),
        top_k=2,
        filters=None,
        collection=None,
    )

    assert len(results) == 2


def test_vector_search_scopes_by_collection(
    db_session, test_user, test_collection
):
    user, _ = test_user
    from app.services.collections import create_collection

    other_collection = create_collection(
        db_session, user, name="Other Collection", slug="other-collection"
    )
    in_scope = _make_chunk(
        db_session,
        test_collection,
        content="in scope",
        chunk_vector=_vec(1.0),
    )
    _make_chunk(
        db_session,
        other_collection,
        content="out of scope",
        chunk_vector=_vec(1.0),
    )

    results = vector_search(
        db_session,
        _vec(1.0),
        top_k=5,
        filters=None,
        collection=str(test_collection.id),
    )

    assert len(results) == 1
    assert results[0].chunk.id == in_scope.id


def test_vector_search_applies_filters(db_session, test_collection, monkeypatch):
    apply_calls: list[dict] = []
    original_apply = _apply_filters

    def spy(stmt, filters):
        apply_calls.append(filters)
        return original_apply(stmt, filters)

    monkeypatch.setattr("app.rag.retrieval._apply_filters", spy)

    filters = {"metadata": {"doc_type": "contract"}}
    vector_search(
        db_session,
        _vec(1.0),
        top_k=5,
        filters=filters,
        collection=None,
    )

    assert apply_calls == [filters]


def test_vector_search_with_metadata_filter(db_session, test_collection):
    match = _make_chunk(
        db_session,
        test_collection,
        content="contract",
        metadata={"doc_type": "contract"},
        chunk_vector=_vec(1.0),
    )
    _make_chunk(
        db_session,
        test_collection,
        content="memo",
        metadata={"doc_type": "memo"},
        chunk_vector=_vec(1.0),
    )

    results = vector_search(
        db_session,
        _vec(1.0),
        top_k=5,
        filters={"metadata": {"doc_type": "contract"}},
        collection=str(test_collection.id),
    )

    assert len(results) == 1
    assert results[0].chunk.id == match.id


def test_vector_search_returns_empty_when_no_rows(db_session):
    results = vector_search(
        db_session,
        [0.0] * 1536,
        top_k=5,
        filters=None,
        collection=None,
    )

    assert results == []


# --- retrieval.py: embed_query / retrieve ---


def test_embed_query_delegates_to_embed_fn():
    called_with: list[str] = []

    def embed_fn(text: str) -> list[float]:
        called_with.append(text)
        return [1.0, 2.0]

    assert embed_query("hello", embed_fn) == [1.0, 2.0]
    assert called_with == ["hello"]


def test_retrieve_embeds_and_searches(db_session, test_collection):
    chunk = _make_chunk(
        db_session,
        test_collection,
        content="Alpha content",
        chunk_vector=_vec(1.0),
    )

    results = retrieve(
        query="Which chunk?",
        top_k=2,
        filters=None,
        collection=str(test_collection.id),
        session=db_session,
        embed_fn=lambda _query: _vec(1.0),
    )

    assert len(results) == 1
    assert results[0].chunk.id == chunk.id
    assert results[0].similarity_score == pytest.approx(1.0, abs=1e-4)


# --- generation.py ---


def test_normalize_chunks_from_retrieved(db_session, test_collection):
    chunk = _make_chunk(db_session, test_collection, content="body")
    retrieved = [_retrieved(chunk, 0.75)]

    normalized = normalize_chunks(retrieved)

    assert len(normalized) == 1
    assert normalized[0].chunk.id == chunk.id
    assert normalized[0].score == 0.75


def test_build_context_orders_by_score_and_tags():
    low = MagicMock()
    low.document_id = uuid4()
    low.chunk_index = 1
    low.content = "less relevant"

    high = MagicMock()
    high.document_id = uuid4()
    high.chunk_index = 0
    high.content = "most relevant"

    context = build_context([_scored(low, 0.3), _scored(high, 0.9)])

    assert context.index("most relevant") < context.index("less relevant")
    assert f"[source: {high.document_id}#0]" in context


def test_build_context_deduplicates_content():
    chunk = MagicMock()
    chunk.document_id = uuid4()
    chunk.chunk_index = 0
    chunk.content = "duplicate text"

    context = build_context([_scored(chunk, 0.9), _scored(chunk, 0.5)])
    assert context.count("duplicate text") == 1


def test_build_context_respects_max_chars():
    chunks = []
    for i in range(5):
        c = MagicMock()
        c.document_id = uuid4()
        c.chunk_index = i
        c.content = f"block-{i}-" + ("x" * 50)
        chunks.append(_scored(c, 1.0 - i * 0.1))

    context = build_context(chunks, max_chars=200)
    assert len(context) <= 200
    assert "block-0" in context


def test_generate_answer_returns_mock_content():
    client = fake_completion_client(answer="The SLA is 99.9%")
    result = generate_answer(
        "What is the SLA?",
        "Context about SLA",
        client,
        completion_model="test-model",
    )
    assert result == "The SLA is 99.9%"


def test_generate_answer_handles_none_content():
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=None))]
    )
    result = generate_answer("q", "ctx", client, completion_model="test-model")
    assert result == ""


def test_answer_with_retrieval_returns_rag_response():
    chunk = MagicMock()
    chunk.id = uuid4()
    chunk.document_id = uuid4()
    chunk.chunk_index = 0
    chunk.content = "SLA is 99.9%"
    scored = [_scored(chunk, 0.95)]
    client = fake_completion_client(answer="The SLA is 99.9%")

    result = answer_with_retrieval(
        "What is the SLA?", scored, client, completion_model="test-model"
    )

    assert result.answer == "The SLA is 99.9%"
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == str(chunk.id)
    assert result.sources[0].score == 0.95
