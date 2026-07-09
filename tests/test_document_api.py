"""Tests for document validation, upload, status, and auth."""

from unittest.mock import patch
from uuid import UUID

import pytest

from app.models import Document
from app.services.collections import create_collection
from app.services.system_user import create_system_user


@patch("app.services.documents.trigger_process_document")
def test_upload_valid_document(mock_trigger, client, auth_headers, db_session, test_user, test_collection) -> None:
    user, _ = test_user
    response = client.post(
        "/documents",
        headers=auth_headers,
        data={"collection_id": str(test_collection.id)},
        files={"file": ("guide.md", b"# Guide\n\nHello world.", "text/markdown")},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["filename"] == "guide.md"
    assert payload["document_id"]

    document = db_session.get(Document, UUID(payload["document_id"]))
    assert document is not None
    assert document.collection_id == test_collection.id
    assert document.status is None
    assert document.content == "# Guide\n\nHello world."
    mock_trigger.assert_called_once_with(payload["document_id"])


@patch("app.services.documents.trigger_process_document")
def test_upload_invalid_document_not_persisted(
    mock_trigger, client, auth_headers, db_session, test_collection
) -> None:
    response = client.post(
        "/documents",
        headers=auth_headers,
        data={"collection_id": str(test_collection.id)},
        files={"file": ("guide.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["accepted"] is False
    assert db_session.query(Document).count() == 0
    mock_trigger.assert_not_called()


def test_upload_requires_auth(client, test_collection) -> None:
    response = client.post(
        "/documents",
        data={"collection_id": str(test_collection.id)},
        files={"file": ("guide.md", b"# Guide", "text/markdown")},
    )
    assert response.status_code == 401


@patch("app.services.documents.trigger_process_document")
def test_upload_duplicate_filename_returns_409(
    mock_trigger, client, auth_headers, db_session, test_user, test_collection
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

    response = client.post(
        "/documents",
        headers=auth_headers,
        data={"collection_id": str(test_collection.id)},
        files={"file": ("guide.md", b"# New", "text/markdown")},
    )

    assert response.status_code == 409
    mock_trigger.assert_not_called()


@patch("app.services.documents.trigger_process_document")
def test_status_returns_queued_before_worker(
    mock_trigger, client, auth_headers, db_session, test_user, test_collection
) -> None:
    document = Document(
        collection_id=test_collection.id,
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

    response = client.get(f"/documents/{document.id}/status", headers=auth_headers)
    assert response.status_code == 404


@patch("app.services.documents.job_status.get_job_status")
def test_status_reads_redis_first(
    mock_redis_status, client, auth_headers, db_session, test_user, test_collection
) -> None:
    document = Document(
        collection_id=test_collection.id,
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
