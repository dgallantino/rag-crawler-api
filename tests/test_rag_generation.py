"""Tests for RAG generation module.

Generation unit tests use mocked completion clients — no live API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from app.models import DocumentChunk
from app.rag.generation import (
    MergedChunk,
    ScoredChunk,
    _strip_leading_overlap,
    answer_with_retrieval,
    build_context,
    generate_answer,
    merge_adjacent_chunks,
    normalize_chunks,
)
from app.rag.retrieval import RetrievedChunk
from tests.rag_helpers import _make_chunk


def fake_completion_client(answer: str = "Mock answer") -> MagicMock:
    mock = MagicMock()

    def create(*, model, messages, **kwargs):
        return MagicMock(choices=[MagicMock(message=MagicMock(content=answer))])

    mock.chat.completions.create = create
    return mock


def _retrieved(chunk: DocumentChunk, similarity_score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, similarity_score=similarity_score)


def _scored(chunk: DocumentChunk, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score)


def _mock_chunk(
    *,
    content: str,
    chunk_index: int = 0,
    document_id=None,
    collection_id=None,
):
    chunk = MagicMock()
    chunk.document_id = document_id if document_id is not None else uuid4()
    chunk.collection_id = collection_id if collection_id is not None else uuid4()
    chunk.chunk_index = chunk_index
    chunk.content = content
    return chunk


# --- generation.py: normalize_chunks ---


def test_normalize_chunks_from_retrieved(db_session, test_collection):
    chunk = _make_chunk(db_session, test_collection, content="body")
    retrieved = [_retrieved(chunk, 0.75)]

    normalized = normalize_chunks(retrieved)

    assert len(normalized) == 1
    assert normalized[0].chunk.id == chunk.id
    assert normalized[0].score == 0.75


# --- generation.py: merge_adjacent_chunks / overlap ---


def test_strip_leading_overlap():
    assert _strip_leading_overlap("hello world", "world peace") == " peace"
    assert _strip_leading_overlap("hello world", "world") == ""
    assert _strip_leading_overlap("abc", "xyz") == "xyz"
    assert _strip_leading_overlap("", "abc") == "abc"


def test_merge_adjacent_chunks_consecutive_same_doc():
    collection_id = uuid4()
    document_id = uuid4()
    chunks = [
        _scored(
            _mock_chunk(
                content="a",
                chunk_index=1,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.4,
        ),
        _scored(
            _mock_chunk(
                content="c",
                chunk_index=3,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.95,
        ),
        _scored(
            _mock_chunk(
                content="b",
                chunk_index=2,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.5,
        ),
    ]

    merged = merge_adjacent_chunks(chunks)

    assert len(merged) == 1
    assert merged[0].chunk_indices == [1, 2, 3]
    assert merged[0].score == 0.95
    assert merged[0].content == "a\n\nb\n\nc"


def test_merged_chunk_content_strips_overlap():
    collection_id = uuid4()
    document_id = uuid4()
    members = [
        _scored(
            _mock_chunk(
                content="hello world",
                chunk_index=0,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.9,
        ),
        _scored(
            _mock_chunk(
                content="world peace",
                chunk_index=1,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.8,
        ),
    ]

    merged = MergedChunk(members=members, score=0.9)
    assert merged.content == "hello world\n\n peace"


def test_merged_chunk_content_no_overlap():
    collection_id = uuid4()
    document_id = uuid4()
    members = [
        _scored(
            _mock_chunk(
                content="a",
                chunk_index=0,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.5,
        ),
        _scored(
            _mock_chunk(
                content="b",
                chunk_index=1,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.4,
        ),
    ]

    merged = MergedChunk(members=members, score=0.5)
    assert merged.content == "a\n\nb"


def test_merged_chunk_content_full_containment_skips_empty():
    collection_id = uuid4()
    document_id = uuid4()
    members = [
        _scored(
            _mock_chunk(
                content="hello world",
                chunk_index=0,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.9,
        ),
        _scored(
            _mock_chunk(
                content="world",
                chunk_index=1,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.7,
        ),
    ]

    merged = MergedChunk(members=members, score=0.9)
    assert merged.content == "hello world"


def test_merge_adjacent_chunks_gap_keeps_separate():
    collection_id = uuid4()
    document_id = uuid4()
    chunks = [
        _scored(
            _mock_chunk(
                content="one",
                chunk_index=1,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.5,
        ),
        _scored(
            _mock_chunk(
                content="three",
                chunk_index=3,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.9,
        ),
    ]

    merged = merge_adjacent_chunks(chunks)

    assert len(merged) == 2
    assert merged[0].chunk_indices == [3]
    assert merged[1].chunk_indices == [1]


def test_merge_adjacent_chunks_different_doc_or_collection():
    collection_id = uuid4()
    doc_a = uuid4()
    doc_b = uuid4()
    other_collection = uuid4()
    chunks = [
        _scored(
            _mock_chunk(
                content="a0",
                chunk_index=0,
                document_id=doc_a,
                collection_id=collection_id,
            ),
            0.5,
        ),
        _scored(
            _mock_chunk(
                content="b1",
                chunk_index=1,
                document_id=doc_b,
                collection_id=collection_id,
            ),
            0.6,
        ),
        _scored(
            _mock_chunk(
                content="a1-other-coll",
                chunk_index=1,
                document_id=doc_a,
                collection_id=other_collection,
            ),
            0.7,
        ),
    ]

    merged = merge_adjacent_chunks(chunks)

    assert len(merged) == 3
    assert all(len(m.chunk_indices) == 1 for m in merged)


def test_merge_adjacent_chunks_orders_by_max_member_score():
    collection_id = uuid4()
    doc_a = uuid4()
    doc_b = uuid4()
    chunks = [
        _scored(
            _mock_chunk(
                content="a1",
                chunk_index=1,
                document_id=doc_a,
                collection_id=collection_id,
            ),
            0.4,
        ),
        _scored(
            _mock_chunk(
                content="a2",
                chunk_index=2,
                document_id=doc_a,
                collection_id=collection_id,
            ),
            0.5,
        ),
        _scored(
            _mock_chunk(
                content="a3",
                chunk_index=3,
                document_id=doc_a,
                collection_id=collection_id,
            ),
            0.95,
        ),
        _scored(
            _mock_chunk(
                content="b0",
                chunk_index=0,
                document_id=doc_b,
                collection_id=collection_id,
            ),
            0.8,
        ),
    ]

    merged = merge_adjacent_chunks(chunks)

    assert len(merged) == 2
    assert merged[0].chunk_indices == [1, 2, 3]
    assert merged[0].score == 0.95
    assert merged[1].chunk_indices == [0]
    assert merged[1].score == 0.8


# --- generation.py: build_context ---


def test_build_context_orders_by_score_and_tags():
    low = _mock_chunk(content="less relevant", chunk_index=1)
    high = _mock_chunk(content="most relevant", chunk_index=0)

    context = build_context([_scored(low, 0.3), _scored(high, 0.9)])

    assert context.index("most relevant") < context.index("less relevant")
    assert f"[source: {high.document_id}#0]" in context


def test_build_context_merges_adjacent_and_tags_range():
    collection_id = uuid4()
    document_id = uuid4()
    chunks = [
        _scored(
            _mock_chunk(
                content="part-1",
                chunk_index=1,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.4,
        ),
        _scored(
            _mock_chunk(
                content="part-2",
                chunk_index=2,
                document_id=document_id,
                collection_id=collection_id,
            ),
            0.9,
        ),
    ]

    context = build_context(chunks)

    assert f"[source: {document_id}#[1,2]]" in context
    assert context.index("part-1") < context.index("part-2")


def test_build_context_deduplicates_content():
    chunk = _mock_chunk(content="duplicate text", chunk_index=0)

    context = build_context([_scored(chunk, 0.9), _scored(chunk, 0.5)])
    assert context.count("duplicate text") == 1


def test_build_context_respects_max_tokens(monkeypatch):
    monkeypatch.setattr(
        "app.rag.generation._token_length",
        lambda text: len(text),
    )
    chunks = []
    for i in range(5):
        c = _mock_chunk(content=f"block-{i}", chunk_index=i)
        chunks.append(_scored(c, 1.0 - i * 0.1))

    # First block alone is ~"[source: uuid#0]\nblock-0" (~50 chars); budget fits one.
    context = build_context(chunks, max_tokens=60)
    assert "block-0" in context
    assert "block-4" not in context
    assert len(context) <= 60


# --- generation.py: generate_answer / answer_with_retrieval ---


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
