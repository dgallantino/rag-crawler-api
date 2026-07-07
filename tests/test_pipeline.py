"""Tests for document processing pipeline."""

from unittest.mock import patch
from uuid import uuid4

import pytest

from app.models import Document, DocumentChunk
from app.rag.pipeline import process_document


@patch("app.rag.pipeline.job_status.delete_job_status")
@patch("app.rag.pipeline.job_status.set_job_step")
@patch("app.rag.pipeline.embeddings.embed_texts")
@patch("app.rag.pipeline.chunking.chunk_markdown")
def test_process_document_success(
    mock_chunk,
    mock_embed,
    mock_set_step,
    mock_delete_status,
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

    from app.rag.chunking import ChunkResult

    mock_chunk.return_value = [
        ChunkResult(content="chunk one", metadata={"section_header": "Pipeline"}),
        ChunkResult(content="chunk two", metadata={"section_header": "Pipeline"}),
    ]
    mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]

    process_document(db_session, str(document.id))

    db_session.refresh(document)
    assert document.status == "success"
    assert document.error_message is None

    chunks = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).all()
    assert len(chunks) == 2
    assert chunks[0].chunk_metadata == {"section_header": "Pipeline"}
    mock_delete_status.assert_called_once()


@patch("app.rag.pipeline.job_status.delete_job_status")
@patch("app.rag.pipeline.job_status.set_job_step")
@patch("app.rag.pipeline.embeddings.embed_texts", side_effect=RuntimeError("embed failed"))
@patch("app.rag.pipeline.chunking.chunk_markdown")
def test_process_document_failure(
    mock_chunk,
    mock_embed,
    mock_set_step,
    mock_delete_status,
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

    from app.rag.chunking import ChunkResult

    mock_chunk.return_value = [ChunkResult(content="chunk", metadata={})]

    with pytest.raises(RuntimeError, match="embed failed"):
        process_document(db_session, str(document.id))

    db_session.refresh(document)
    assert document.status == "failed"
    assert document.error_message == "embed failed"
    mock_delete_status.assert_called_once()


def test_process_document_missing_is_noop(db_session) -> None:
    process_document(db_session, str(uuid4()))
