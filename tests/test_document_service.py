from unittest.mock import patch
from uuid import uuid4

import pytest

from app.models import Document
from app.services.collections import create_collection
from app.services.documents import (
    DocumentConflictError,
    DocumentNotFoundError,
    create_document_upload,
    get_document_status,
    validate_markdown_upload,
)
from app.services.system_user import create_system_user


def test_validate_rejects_non_md() -> None:
    result = validate_markdown_upload("readme.txt", b"hello")
    assert result.valid is False
    assert result.reason == "Only .md files are accepted"


def test_validate_rejects_empty() -> None:
    result = validate_markdown_upload("doc.md", b"   \n  ")
    assert result.valid is False
    assert result.reason == "File content is empty"


def test_validate_rejects_binary() -> None:
    result = validate_markdown_upload("doc.md", b"hello\x00world")
    assert result.valid is False
    assert result.reason == "File contains binary content"


def test_validate_accepts_markdown() -> None:
    result = validate_markdown_upload("doc.md", b"# Title\n\nSome content.")
    assert result.valid is True


@patch("app.services.documents.trigger_process_document")
def test_create_document_upload_persists_and_triggers(
    mock_trigger, db_session, test_user, test_collection
) -> None:
    document = create_document_upload(db_session, test_collection, "guide.md", "# Guide\n\nHello world.")

    assert document.collection_id == test_collection.id
    assert document.title == "guide.md"
    assert document.url == "file://guide.md"
    assert document.content == "# Guide\n\nHello world."
    assert document.status is None

    persisted = db_session.get(Document, document.id)
    assert persisted is not None
    assert persisted.content == "# Guide\n\nHello world."

    mock_trigger.assert_called_once_with(str(document.id))


@patch("app.services.documents.trigger_process_document")
def test_create_document_upload_raises_on_duplicate(
    mock_trigger, db_session, test_user, test_collection
) -> None:
    db_session.add(
        Document(
            collection_id=test_collection.id,
            url="file://guide.md",
            title="guide.md",
            content="existing",
        )
    )
    db_session.commit()

    with pytest.raises(DocumentConflictError, match="guide.md"):
        create_document_upload(db_session, test_collection, "guide.md", "# New content")

    assert db_session.query(Document).count() == 1
    mock_trigger.assert_not_called()


def test_get_document_status_raises_when_not_found(db_session, test_user) -> None:
    user, _ = test_user

    with pytest.raises(DocumentNotFoundError):
        get_document_status(db_session, user, uuid4())


def test_get_document_status_raises_for_other_user(db_session, test_user) -> None:
    user, _ = test_user
    other_user, _ = create_system_user(db_session, name="Other Tenant")
    other_collection = create_collection(db_session, other_user, name="Other Coll", slug="other-coll")
    document = Document(
        collection_id=other_collection.id,
        url="file://other.md",
        title="other.md",
        content="# Other",
    )
    db_session.add(document)
    db_session.commit()

    with pytest.raises(DocumentNotFoundError):
        get_document_status(db_session, user, document.id)


def test_get_document_status_returns_queued_without_redis(db_session, test_user, test_collection) -> None:
    user, _ = test_user
    document = Document(
        collection_id=test_collection.id,
        url="file://pending.md",
        title="pending.md",
        content="# Pending",
    )
    db_session.add(document)
    db_session.commit()

    response = get_document_status(db_session, user, document.id)

    assert response.document_id == document.id
    assert response.status == "queued"
    assert response.step is None
    assert response.steps is None
    assert response.error_message is None


def test_get_document_status_returns_db_status_without_redis(db_session, test_user, test_collection) -> None:
    user, _ = test_user
    document = Document(
        collection_id=test_collection.id,
        url="file://done.md",
        title="done.md",
        content="# Done",
        status="success",
    )
    db_session.add(document)
    db_session.commit()

    response = get_document_status(db_session, user, document.id)

    assert response.document_id == document.id
    assert response.status == "success"
    assert response.step is None
    assert response.steps is None


@patch("app.services.documents.job_status.get_job_status")
def test_get_document_status_reads_redis_when_available(
    mock_redis_status, db_session, test_user, test_collection
) -> None:
    user, _ = test_user
    document = Document(
        collection_id=test_collection.id,
        url="file://active.md",
        title="active.md",
        content="# Active",
        status=None,
        error_message="previous warning",
    )
    db_session.add(document)
    db_session.commit()

    mock_redis_status.return_value = {
        "step": "embedding",
        "steps": {
            "chunking": "completed",
            "embedding": "in_progress",
            "storing": "pending",
        },
    }

    response = get_document_status(db_session, user, document.id)

    assert response.document_id == document.id
    assert response.status == "processing"
    assert response.step == "embedding"
    assert response.steps == {
        "chunking": "completed",
        "embedding": "in_progress",
        "storing": "pending",
    }
    assert response.error_message == "previous warning"


def test_get_document_status_includes_error_message(db_session, test_user, test_collection) -> None:
    user, _ = test_user
    document = Document(
        collection_id=test_collection.id,
        url="file://failed.md",
        title="failed.md",
        content="# Failed",
        status="failed",
        error_message="embedding failed",
    )
    db_session.add(document)
    db_session.commit()

    response = get_document_status(db_session, user, document.id)

    assert response.document_id == document.id
    assert response.status == "failed"
    assert response.error_message == "embedding failed"
