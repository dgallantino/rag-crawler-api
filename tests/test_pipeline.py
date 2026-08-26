"""Tests for document processing pipeline."""

from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.models import Document, DocumentChunk
from app.rag.events import DocumentStatusEvent
from app.rag.processor import ChunkResult, MarkdownProcessor


def _make_processor() -> MarkdownProcessor:
    return MarkdownProcessor(
        Mock(),
        "test-model",
        chunk_max_tokens=500,
        chunk_min_tokens=300,
        chunk_overlap_percent=10,
    )


def test_process_document_success(
    db_session,
    test_user,
    test_collection,
) -> None:
    document = Document(
        collection_id=test_collection.id,
        url="file://pipeline.md",
        title="pipeline.md",
        content="# Pipeline\n\nContent.",
    )
    db_session.add(document)
    db_session.commit()

    processor = _make_processor()
    processor.chunk = Mock(
        return_value=[
            ChunkResult(content="chunk one", metadata={"section_header": "Pipeline"}),
            ChunkResult(content="chunk two", metadata={"section_header": "Pipeline"}),
        ]
    )
    processor.embed_texts = Mock(return_value=[[0.1] * 1536, [0.2] * 1536])

    on_status = Mock()
    processor.process_document(db_session, str(document.id), on_status=on_status)

    db_session.refresh(document)
    assert document.status == "success"
    assert document.error_message is None

    processor.embed_texts.assert_called_once_with(
        [
            "Title: pipeline.md\nHeading: Pipeline\n\nchunk one",
            "Title: pipeline.md\nHeading: Pipeline\n\nchunk two",
        ]
    )

    chunks = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).all()
    assert len(chunks) == 2
    assert chunks[0].content == "chunk one"
    assert chunks[1].content == "chunk two"
    assert chunks[0].chunk_metadata == {"section_header": "Pipeline"}
    assert chunks[0].collection_id == test_collection.id

    assert on_status.call_args_list == [
        ((DocumentStatusEvent(str(document.id), "chunking", "in_progress"),),),
        ((DocumentStatusEvent(str(document.id), "chunking", "completed"),),),
        ((DocumentStatusEvent(str(document.id), "embedding", "in_progress"),),),
        ((DocumentStatusEvent(str(document.id), "embedding", "completed"),),),
        ((DocumentStatusEvent(str(document.id), "storing", "in_progress"),),),
        ((DocumentStatusEvent(str(document.id), "storing", "completed"),),),
    ]


def test_process_document_failure(
    db_session,
    test_user,
    test_collection,
) -> None:
    document = Document(
        collection_id=test_collection.id,
        url="file://fail.md",
        title="fail.md",
        content="# Fail\n\nContent.",
    )
    db_session.add(document)
    db_session.commit()

    processor = _make_processor()
    processor.chunk = Mock(return_value=[ChunkResult(content="chunk", metadata={})])
    processor.embed_texts = Mock(side_effect=RuntimeError("embed failed"))

    on_status = Mock()
    with pytest.raises(RuntimeError, match="embed failed"):
        processor.process_document(db_session, str(document.id), on_status=on_status)

    db_session.refresh(document)
    assert document.status == "failed"
    assert document.error_message == "embed failed"

    assert on_status.call_args_list == [
        ((DocumentStatusEvent(str(document.id), "chunking", "in_progress"),),),
        ((DocumentStatusEvent(str(document.id), "chunking", "completed"),),),
        ((DocumentStatusEvent(str(document.id), "embedding", "in_progress"),),),
        ((DocumentStatusEvent(str(document.id), "failed", "completed"),),),
    ]


def test_process_document_missing_is_noop(db_session) -> None:
    processor = _make_processor()
    on_status = Mock()
    processor.process_document(db_session, str(uuid4()), on_status=on_status)
    on_status.assert_not_called()


def test_process_document_commits_only_after_success(
    db_session,
    test_collection,
) -> None:
    document = Document(
        collection_id=test_collection.id,
        url="file://txn.md",
        title="txn.md",
        content="# Txn\n\nContent.",
    )
    db_session.add(document)
    db_session.commit()

    processor = _make_processor()
    processor.chunk = Mock(
        return_value=[ChunkResult(content="chunk", metadata={})]
    )
    processor.embed_texts = Mock(return_value=[[0.1] * 1536])

    commit_calls = 0
    original_commit = db_session.commit

    def counting_commit() -> None:
        nonlocal commit_calls
        commit_calls += 1
        original_commit()

    db_session.commit = counting_commit  # type: ignore[method-assign]

    processor.process_document(db_session, str(document.id), on_status=Mock())

    assert commit_calls == 1
    db_session.refresh(document)
    assert document.status == "success"


def test_process_document_empty_chunks_fails_without_deleting_existing(
    db_session,
    test_collection,
) -> None:
    document = Document(
        collection_id=test_collection.id,
        url="file://empty-chunks.md",
        title="empty-chunks.md",
        content="# Existing\n\nContent.",
        status="success",
    )
    db_session.add(document)
    db_session.commit()

    existing = DocumentChunk(
        document_id=document.id,
        collection_id=test_collection.id,
        chunk_index=0,
        content="keep me",
        chunk_vector=[0.1] * 1536,
    )
    db_session.add(existing)
    db_session.commit()

    processor = _make_processor()
    processor.chunk = Mock(return_value=[])
    on_status = Mock()

    with pytest.raises(ValueError, match="zero chunks"):
        processor.process_document(db_session, str(document.id), on_status=on_status)

    db_session.refresh(document)
    assert document.status == "failed"
    assert "zero chunks" in (document.error_message or "")

    chunks = (
        db_session.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document.id)
        .all()
    )
    assert len(chunks) == 1
    assert chunks[0].content == "keep me"


def test_process_document_skips_store_when_superseded(
    db_session,
    test_collection,
) -> None:
    document = Document(
        collection_id=test_collection.id,
        url="file://race.md",
        title="race.md",
        content="version-b",
    )
    db_session.add(document)
    db_session.commit()

    existing = DocumentChunk(
        document_id=document.id,
        collection_id=test_collection.id,
        chunk_index=0,
        content="chunk-c",
        chunk_vector=[0.9] * 1536,
    )
    db_session.add(existing)
    db_session.commit()

    processor = _make_processor()

    def chunk_and_supersede(content: str) -> list[ChunkResult]:
        document.content = "version-c"
        document.title = "updated.md"
        db_session.commit()
        return [ChunkResult(content="chunk-b", metadata={})]

    processor.chunk = Mock(side_effect=chunk_and_supersede)
    processor.embed_texts = Mock(return_value=[[0.1] * 1536])

    on_status = Mock()
    processor.process_document(db_session, str(document.id), on_status=on_status)

    db_session.refresh(document)
    assert document.content == "version-c"
    assert document.title == "updated.md"

    chunks = (
        db_session.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )
    assert len(chunks) == 1
    assert chunks[0].content == "chunk-c"

    storing_events = [
        call.args[0]
        for call in on_status.call_args_list
        if call.args[0].step == "storing"
    ]
    assert storing_events == []
    assert document.status is None
    failed_events = [
        call.args[0]
        for call in on_status.call_args_list
        if call.args[0].step == "failed"
    ]
    assert failed_events == []


def test_process_document_failure_does_not_mark_failed_when_superseded(
    db_session,
    test_collection,
) -> None:
    document = Document(
        collection_id=test_collection.id,
        url="file://race-fail.md",
        title="race-fail.md",
        content="version-b",
        status="success",
    )
    db_session.add(document)
    db_session.commit()

    processor = _make_processor()

    def chunk_and_supersede(content: str) -> list[ChunkResult]:
        document.content = "version-c"
        document.status = None
        db_session.commit()
        return [ChunkResult(content="chunk-b", metadata={})]

    processor.chunk = Mock(side_effect=chunk_and_supersede)
    processor.embed_texts = Mock(side_effect=RuntimeError("embed failed"))

    on_status = Mock()
    with pytest.raises(RuntimeError, match="embed failed"):
        processor.process_document(db_session, str(document.id), on_status=on_status)

    db_session.refresh(document)
    assert document.content == "version-c"
    assert document.status is None
    assert document.error_message is None

    failed_events = [
        call.args[0]
        for call in on_status.call_args_list
        if call.args[0].step == "failed"
    ]
    assert failed_events == []
