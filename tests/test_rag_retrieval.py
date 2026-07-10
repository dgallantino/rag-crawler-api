"""Tests for RAG retrieval, rerank, and stitch modules.

All tests use mocked embed_fn and completion_client — no live API calls.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.models import Document, DocumentChunk
from app.rag.rerank import (
    _assign_rank_scores,
    _build_rerank_prompt,
    _parse_rerank_response,
    rerank,
)
from app.rag.retrieval import (
    ScoredChunk,
    _apply_filters,
    _distance_to_score,
    retrieve,
)
from app.rag.stitch import build_context, generate_answer, stitch


def fake_embed_fn(_query: str) -> list[float]:
    return [0.0] * 1536


def fake_completion_client(
    rerank_ids: list[str] | None = None,
    answer: str = "Mock answer",
) -> MagicMock:
    mock = MagicMock()

    def create(*, model, messages, **kwargs):
        content = messages[-1]["content"]
        if rerank_ids is not None and "Rank the following" in content:
            body = json.dumps(rerank_ids)
        else:
            body = answer
        return MagicMock(choices=[MagicMock(message=MagicMock(content=body))])

    mock.chat.completions.create = create
    return mock


def _make_chunk(
    db_session,
    collection,
    *,
    content: str = "chunk content",
    metadata: dict | None = None,
    created_at: datetime | None = None,
    chunk_vector: list[float] | None = None,
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
        chunk_index=0,
        content=content,
        chunk_metadata=metadata,
        created_at=created_at or datetime(2024, 6, 15, 12, 0, 0),
        chunk_vector=chunk_vector or ([0.1] * 1536),
    )
    db_session.add(chunk)
    db_session.commit()
    return chunk


def _scored(chunk: DocumentChunk, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score)


# --- retrieval.py ---


def test_distance_to_score_clamps():
    assert _distance_to_score(0.0) == 1.0
    assert _distance_to_score(1.0) == 0.0
    assert _distance_to_score(1.5) == 0.0
    assert _distance_to_score(-0.2) == 1.0


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


def test_apply_filters_date_range(db_session, test_collection):
    early = _make_chunk(
        db_session,
        test_collection,
        content="early",
        created_at=datetime(2024, 1, 10, 0, 0, 0),
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

    assert len(rows) == 0  # early is Jan, late is Dec — neither in June

    stmt = select(DocumentChunk)
    stmt = _apply_filters(stmt, {"date_range": {"after": date(2024, 1, 1)}})
    rows = db_session.execute(stmt).scalars().all()
    assert len(rows) == 2

    stmt = select(DocumentChunk)
    stmt = _apply_filters(stmt, {"date_range": {"before": date(2024, 2, 1)}})
    rows = db_session.execute(stmt).scalars().all()
    assert len(rows) == 1
    assert rows[0].id == early.id


def test_apply_filters_owner_ref_skipped(db_session, test_collection):
    chunk = _make_chunk(db_session, test_collection, content="owned")
    stmt = select(DocumentChunk)
    stmt = _apply_filters(stmt, {"owner_ref": "user-123"})
    rows = db_session.execute(stmt).scalars().all()
    assert len(rows) == 1
    assert rows[0].id == chunk.id


def test_collection_uuid_filter(db_session, test_collection, test_user):
    """Collection scoping uses UUID cast (pgvector search requires Postgres)."""
    from uuid import UUID

    from app.services.collections import create_collection

    user, _ = test_user
    other_collection = create_collection(db_session, user, name="Other", slug="other-col")
    in_collection = _make_chunk(db_session, test_collection, content="in scope")
    _make_chunk(db_session, other_collection, content="out of scope")

    stmt = select(DocumentChunk).where(
        DocumentChunk.collection_id == UUID(str(test_collection.id))
    )
    rows = db_session.execute(stmt).scalars().all()

    assert len(rows) == 1
    assert rows[0].id == in_collection.id


# --- rerank.py ---


def test_build_rerank_prompt_includes_ids_and_query():
    chunk = MagicMock()
    chunk.id = uuid4()
    chunk.content = "Some content about SLAs"
    candidates = [_scored(chunk)]

    prompt = _build_rerank_prompt("What is the SLA?", candidates)

    assert "What is the SLA?" in prompt
    assert str(chunk.id) in prompt
    assert "Some content about SLAs" in prompt
    assert "JSON array" in prompt


def test_parse_rerank_response_json_array():
    response = MagicMock(
        choices=[MagicMock(message=MagicMock(content='["id-1", "id-2"]'))]
    )
    chunk = MagicMock()
    chunk.id = "fallback"
    candidates = [_scored(chunk)]

    assert _parse_rerank_response(response, candidates) == ["id-1", "id-2"]


def test_parse_rerank_response_strips_code_fence():
    response = MagicMock(
        choices=[
            MagicMock(message=MagicMock(content='```json\n["a", "b"]\n```'))
        ]
    )
    chunk = MagicMock()
    chunk.id = "x"
    candidates = [_scored(chunk)]

    assert _parse_rerank_response(response, candidates) == ["a", "b"]


def test_parse_rerank_response_fallback_on_invalid_json():
    chunk_a = MagicMock()
    chunk_a.id = uuid4()
    chunk_b = MagicMock()
    chunk_b.id = uuid4()
    candidates = [_scored(chunk_a), _scored(chunk_b)]

    response = MagicMock(choices=[MagicMock(message=MagicMock(content="not json"))])
    result = _parse_rerank_response(response, candidates)

    assert result == [str(chunk_a.id), str(chunk_b.id)]


def test_assign_rank_scores():
    chunks = [MagicMock(), MagicMock(), MagicMock()]
    scored = [_scored(c, 0.5) for c in chunks]
    ranked = _assign_rank_scores(scored)

    assert ranked[0].score == 1.0
    assert ranked[-1].score == 0.0
    assert len(ranked) == 3


def test_rerank_reorders_by_model(db_session, test_collection):
    first = _make_chunk(db_session, test_collection, content="first")
    second = _make_chunk(db_session, test_collection, content="second")
    third = _make_chunk(db_session, test_collection, content="third")

    candidates = [
        _scored(first, 0.9),
        _scored(second, 0.8),
        _scored(third, 0.7),
    ]
    rerank_order = [str(third.id), str(first.id), str(second.id)]
    client = fake_completion_client(rerank_ids=rerank_order)

    result = rerank(
        "query",
        candidates,
        client,
        top_k=2,
        completion_model="test-model",
    )

    assert len(result) == 2
    assert result[0].chunk.id == third.id
    assert result[1].chunk.id == first.id
    assert result[0].score == 1.0


def test_rerank_fallback_on_client_error(db_session, test_collection):
    chunk = _make_chunk(db_session, test_collection, content="only")
    candidates = [_scored(chunk, 0.9)]

    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("API down")

    result = rerank(
        "query",
        candidates,
        client,
        top_k=1,
        completion_model="test-model",
    )

    assert len(result) == 1
    assert result[0].chunk.id == chunk.id
    assert result[0].score == 0.9


# --- stitch.py ---


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

    dup_a = _scored(chunk, 0.9)
    dup_b = _scored(chunk, 0.5)

    context = build_context([dup_a, dup_b])
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


def test_stitch_returns_rag_response(db_session, test_collection):
    chunk = _make_chunk(db_session, test_collection, content="SLA is 99.9%")
    scored = [_scored(chunk, 0.95)]
    client = fake_completion_client(answer="The SLA is 99.9%")

    result = stitch("What is the SLA?", scored, client, completion_model="test-model")

    assert result.answer == "The SLA is 99.9%"
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == str(chunk.id)
    assert result.sources[0].score == 0.95


def test_retrieve_end_to_end(db_session, test_collection, monkeypatch):
    chunk_a = _make_chunk(db_session, test_collection, content="Alpha content")
    chunk_b = _make_chunk(db_session, test_collection, content="Beta content")

    def mock_vector_search(session, query_vector, *, top_k, filters, collection):
        return [
            _scored(chunk_a, 0.9),
            _scored(chunk_b, 0.7),
        ][:top_k]

    monkeypatch.setattr("app.rag.retrieval.vector_search", mock_vector_search)

    client = fake_completion_client(
        rerank_ids=[str(chunk_b.id), str(chunk_a.id)],
        answer="Beta is the answer",
    )

    result = retrieve(
        query="Which chunk?",
        top_k=1,
        filters=None,
        rerank=True,
        collection=str(test_collection.id),
        session=db_session,
        embed_fn=fake_embed_fn,
        completion_client=client,
        completion_model="test-model",
    )

    assert result.answer == "Beta is the answer"
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == str(chunk_b.id)
