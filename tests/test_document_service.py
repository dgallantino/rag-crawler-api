from unittest.mock import patch
from uuid import uuid4

import pytest

from app.models import Document
from app.services.collections import CollectionNotFoundError, create_collection
from app.services.documents import (
    DocumentConflictError,
    DocumentNotFoundError,
    DocumentValidationError,
    create_document_upload,
    delete_document,
    get_document,
    get_document_status,
    list_documents,
    update_document,
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


@patch("app.services.documents.trigger_process_document")
def test_create_document_upload_rolls_back_when_enqueue_fails(
    mock_trigger, db_session, test_user, test_collection
) -> None:
    from app.services.triggers import QueueEnqueueError

    mock_trigger.side_effect = QueueEnqueueError()

    with pytest.raises(QueueEnqueueError, match="Failed to queue document processing"):
        create_document_upload(db_session, test_collection, "guide.md", "# Guide")

    assert db_session.query(Document).count() == 0
    mock_trigger.assert_called_once()


@patch("app.services.documents.trigger_process_document")
def test_update_document_marks_failed_when_enqueue_fails(
    mock_trigger, db_session, test_user, test_collection
) -> None:
    from app.services.triggers import QueueEnqueueError

    document = create_document_upload(db_session, test_collection, "guide.md", "# Guide")
    mock_trigger.reset_mock()
    document.status = "success"
    db_session.commit()

    mock_trigger.side_effect = QueueEnqueueError()

    with pytest.raises(QueueEnqueueError, match="Failed to queue document processing"):
        update_document(db_session, test_user, document.id, content="# Revised")

    db_session.refresh(document)
    assert document.content == "# Revised"
    assert document.status == "failed"
    assert document.error_message == "Failed to queue document processing"
    mock_trigger.assert_called_once_with(str(document.id))


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


def test_list_documents_empty(db_session, test_user) -> None:
    user, _ = test_user
    assert list_documents(db_session, user) == []


@patch("app.services.documents.trigger_process_document")
def test_list_documents_populated(
    mock_trigger, db_session, test_user, test_collection
) -> None:
    user, _ = test_user
    first = create_document_upload(db_session, test_collection, "a.md", "# A")
    second = create_document_upload(db_session, test_collection, "b.md", "# B")

    result = list_documents(db_session, user)
    assert {doc.id for doc in result} == {first.id, second.id}


@patch("app.services.documents.trigger_process_document")
def test_list_documents_scoped_to_user(
    mock_trigger, db_session, test_user, test_collection
) -> None:
    user, _ = test_user
    create_document_upload(db_session, test_collection, "mine.md", "# Mine")

    other_user, _ = create_system_user(db_session, name="Other Tenant")
    other_collection = create_collection(
        db_session, other_user, name="Other Coll", slug="other-coll"
    )
    create_document_upload(db_session, other_collection, "theirs.md", "# Theirs")

    result = list_documents(db_session, user)
    assert [doc.title for doc in result] == ["mine.md"]


@patch("app.services.documents.trigger_process_document")
def test_list_documents_filters_by_collection_id(
    mock_trigger, db_session, test_user, test_collection
) -> None:
    user, _ = test_user
    other = create_collection(db_session, user, name="Other", slug="other")
    keep = create_document_upload(db_session, test_collection, "keep.md", "# Keep")
    create_document_upload(db_session, other, "skip.md", "# Skip")

    result = list_documents(db_session, user, collection_id=test_collection.id)
    assert [doc.id for doc in result] == [keep.id]


@patch("app.services.documents.trigger_process_document")
def test_list_documents_filters_by_collection_slug(
    mock_trigger, db_session, test_user, test_collection
) -> None:
    user, _ = test_user
    other = create_collection(db_session, user, name="Other", slug="other")
    create_document_upload(db_session, test_collection, "keep.md", "# Keep")
    create_document_upload(db_session, other, "skip.md", "# Skip")

    result = list_documents(db_session, user, collection_slug="other")
    assert [doc.title for doc in result] == ["skip.md"]


def test_list_documents_rejects_both_collection_identifiers(
    db_session, test_user, test_collection
) -> None:
    user, _ = test_user
    with pytest.raises(ValueError, match="at most one"):
        list_documents(
            db_session,
            user,
            collection_id=test_collection.id,
            collection_slug="test-collection",
        )


def test_list_documents_unknown_collection(db_session, test_user) -> None:
    user, _ = test_user
    with pytest.raises(CollectionNotFoundError):
        list_documents(db_session, user, collection_slug="missing")


def test_get_document_not_found(db_session, test_user) -> None:
    user, _ = test_user
    with pytest.raises(DocumentNotFoundError):
        get_document(db_session, user, uuid4())


def test_get_document_rejects_other_user(db_session, test_user) -> None:
    user, _ = test_user
    other_user, _ = create_system_user(db_session, name="Other Tenant")
    other_collection = create_collection(
        db_session, other_user, name="Other Coll", slug="other-coll"
    )
    document = Document(
        collection_id=other_collection.id,
        url="file://other.md",
        title="other.md",
        content="# Other",
    )
    db_session.add(document)
    db_session.commit()

    with pytest.raises(DocumentNotFoundError):
        get_document(db_session, user, document.id)


@patch("app.services.documents.trigger_process_document")
def test_update_document_title_reindexes_and_keeps_url(
    mock_trigger, db_session, test_user, test_collection
) -> None:
    user, _ = test_user
    document = create_document_upload(db_session, test_collection, "guide.md", "# Guide")
    mock_trigger.reset_mock()
    document.status = "success"
    db_session.commit()

    updated = update_document(db_session, user, document.id, title="Renamed Guide")

    assert updated.title == "Renamed Guide"
    assert updated.url == "file://guide.md"
    assert updated.content == "# Guide"
    assert updated.status == "success"
    mock_trigger.assert_called_once_with(str(document.id))


@patch("app.services.documents.trigger_process_document")
def test_update_document_content_reindexes_and_resets_status(
    mock_trigger, db_session, test_user, test_collection
) -> None:
    user, _ = test_user
    document = create_document_upload(db_session, test_collection, "guide.md", "# Guide")
    mock_trigger.reset_mock()
    document.status = "success"
    document.error_message = "old"
    db_session.commit()

    updated = update_document(db_session, user, document.id, content="# Revised")

    assert updated.content == "# Revised"
    assert updated.title == "guide.md"
    assert updated.status is None
    assert updated.error_message is None
    mock_trigger.assert_called_once_with(str(document.id))


@patch("app.services.documents.trigger_process_document")
def test_update_document_rejects_empty_content(
    mock_trigger, db_session, test_user, test_collection
) -> None:
    user, _ = test_user
    document = create_document_upload(db_session, test_collection, "guide.md", "# Guide")
    mock_trigger.reset_mock()

    with pytest.raises(DocumentValidationError, match="empty"):
        update_document(db_session, user, document.id, content="   \n")

    mock_trigger.assert_not_called()
    persisted = db_session.get(Document, document.id)
    assert persisted is not None
    assert persisted.content == "# Guide"


def test_update_document_not_found(db_session, test_user) -> None:
    user, _ = test_user
    with pytest.raises(DocumentNotFoundError):
        update_document(db_session, user, uuid4(), title="Nope")


@patch("app.services.documents.trigger_process_document")
def test_delete_document(mock_trigger, db_session, test_user, test_collection) -> None:
    user, _ = test_user
    document = create_document_upload(db_session, test_collection, "guide.md", "# Guide")
    document_id = document.id

    delete_document(db_session, user, document_id)

    assert db_session.get(Document, document_id) is None
    with pytest.raises(DocumentNotFoundError):
        get_document(db_session, user, document_id)


@patch("app.services.documents.trigger_process_document")
def test_delete_document_removes_chunks(
    mock_trigger, db_session, test_user, test_collection
) -> None:
    from app.models import DocumentChunk
    from tests.rag_helpers import _make_chunk

    user, _ = test_user
    chunk = _make_chunk(db_session, test_collection, content="chunk")
    document_id = chunk.document_id
    chunk_id = chunk.id

    delete_document(db_session, user, document_id)

    assert db_session.get(Document, document_id) is None
    assert db_session.get(DocumentChunk, chunk_id) is None


def test_delete_document_not_found(db_session, test_user) -> None:
    user, _ = test_user
    with pytest.raises(DocumentNotFoundError):
        delete_document(db_session, user, uuid4())


def test_delete_document_rejects_other_user(db_session, test_user) -> None:
    user, _ = test_user
    other_user, _ = create_system_user(db_session, name="Other Tenant")
    other_collection = create_collection(
        db_session, other_user, name="Other Coll", slug="other-coll"
    )
    document = Document(
        collection_id=other_collection.id,
        url="file://other.md",
        title="other.md",
        content="# Other",
    )
    db_session.add(document)
    db_session.commit()

    with pytest.raises(DocumentNotFoundError):
        delete_document(db_session, user, document.id)
    assert db_session.get(Document, document.id) is not None
