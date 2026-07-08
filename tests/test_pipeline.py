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
        chunk_overlap_percent=10,
    )


def test_process_document_success(
    db_session,
    test_user,
) -> None:
    user, _ = test_user
    document = Document(
        system_user_id=user.id,
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

    chunks = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).all()
    assert len(chunks) == 2
    assert chunks[0].chunk_metadata == {"section_header": "Pipeline"}

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
) -> None:
    user, _ = test_user
    document = Document(
        system_user_id=user.id,
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
