"""Tests for document validation, upload, status, and auth."""

from io import BytesIO
from unittest.mock import patch
from uuid import UUID

import pytest

from app.models import Document
from app.rag.events import DocumentStatusEvent
from app.services.documents import validate_markdown_upload
from app.services.system_user import create_system_user
from app.services.job_status import handle_document_status_event



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
def test_upload_valid_document(mock_trigger, client, auth_headers, db_session, test_user) -> None:
    user, _ = test_user
    response = client.post(
        "/documents",
        headers=auth_headers,
        files={"file": ("guide.md", b"# Guide\n\nHello world.", "text/markdown")},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["filename"] == "guide.md"
    assert payload["document_id"]

    document = db_session.get(Document, UUID(payload["document_id"]))
    assert document is not None
    assert document.system_user_id == user.id
    assert document.status is None
    assert document.content == "# Guide\n\nHello world."
    mock_trigger.assert_called_once_with(payload["document_id"])


@patch("app.services.documents.trigger_process_document")
def test_upload_invalid_document_not_persisted(
    mock_trigger, client, auth_headers, db_session
) -> None:
    response = client.post(
        "/documents",
        headers=auth_headers,
        files={"file": ("guide.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["accepted"] is False
    assert db_session.query(Document).count() == 0
    mock_trigger.assert_not_called()


def test_upload_requires_auth(client) -> None:
    response = client.post(
        "/documents",
        files={"file": ("guide.md", b"# Guide", "text/markdown")},
    )
    assert response.status_code == 401


@patch("app.services.documents.trigger_process_document")
def test_upload_duplicate_filename_returns_409(
    mock_trigger, client, auth_headers, db_session, test_user
) -> None:
    user, _ = test_user
    db_session.add(
        Document(
            system_user_id=user.id,
            url="file://guide.md",
            title="guide.md",
            content="existing",
        )
    )
    db_session.commit()

    response = client.post(
        "/documents",
        headers=auth_headers,
        files={"file": ("guide.md", b"# New", "text/markdown")},
    )

    assert response.status_code == 409
    mock_trigger.assert_not_called()


@patch("app.services.documents.trigger_process_document")
def test_status_returns_queued_before_worker(
    mock_trigger, client, auth_headers, db_session, test_user
) -> None:
    user, _ = test_user
    document = Document(
        system_user_id=user.id,
        url="file://pending.md",
        title="pending.md",
        content="# Pending",
    )
    db_session.add(document)
    db_session.commit()

    response = client.get(f"/documents/{document.id}/status", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_status_not_found_for_other_tenant(client, auth_headers, db_session, test_user) -> None:
    user, _ = test_user
    other_user, _ = create_system_user(db_session, name="Other Tenant")
    document = Document(
        system_user_id=other_user.id,
        url="file://other.md",
        title="other.md",
        content="# Other",
    )
    db_session.add(document)
    db_session.commit()

    response = client.get(f"/documents/{document.id}/status", headers=auth_headers)
    assert response.status_code == 404


@patch("app.services.documents.job_status.get_job_status")
def test_status_reads_redis_first(
    mock_redis_status, client, auth_headers, db_session, test_user
) -> None:
    user, _ = test_user
    document = Document(
        system_user_id=user.id,
        url="file://active.md",
        title="active.md",
        content="# Active",
        status="processing",
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

    response = client.get(f"/documents/{document.id}/status", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["step"] == "embedding"
    assert payload["steps"]["embedding"] == "in_progress"





"""Tests for document status event handlers."""

@patch("app.services.job_status.delete_job_status")
@patch("app.services.job_status.set_job_step")
def test_handle_document_status_event_in_progress(mock_set_step, mock_delete_status) -> None:
    event = DocumentStatusEvent("doc-1", "chunking", "in_progress")
    handle_document_status_event(event)

    mock_set_step.assert_called_once_with("doc-1", "chunking", "in_progress")
    mock_delete_status.assert_not_called()


@patch("app.services.job_status.delete_job_status")
@patch("app.services.job_status.set_job_step")
def test_handle_document_status_event_success_terminal(mock_set_step, mock_delete_status) -> None:
    event = DocumentStatusEvent("doc-1", "storing", "completed")
    handle_document_status_event(event)

    mock_set_step.assert_called_once_with("doc-1", "storing", "completed")
    mock_delete_status.assert_called_once_with("doc-1")


@patch("app.services.job_status.delete_job_status")
@patch("app.services.job_status.set_job_step")
def test_handle_document_status_event_failure_terminal(mock_set_step, mock_delete_status) -> None:
    event = DocumentStatusEvent("doc-1", "failed", "completed")
    handle_document_status_event(event)

    mock_set_step.assert_not_called()
    mock_delete_status.assert_called_once_with("doc-1")