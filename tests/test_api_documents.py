"""HTTP tests for /v1/documents."""

from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.services.system_user import create_system_user

JSON_BODY = {
    "content": "# Hello\n\nWorld.",
    "filename": "hello.md",
    "collection_slug": "test-collection",
}


@patch("app.services.documents.trigger_process_document")
def test_upload_json_list_get_status_patch_delete(
    mock_trigger, client: TestClient, auth_headers, test_collection
) -> None:
    uploaded = client.post("/v1/documents/json", json=JSON_BODY, headers=auth_headers)
    assert uploaded.status_code == 202
    body = uploaded.json()
    assert body["accepted"] is True
    assert body["filename"] == "hello.md"
    document_id = body["document_id"]
    mock_trigger.assert_called_once_with(document_id)

    listed = client.get("/v1/documents", headers=auth_headers)
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["id"] == document_id
    assert listed.json()[0]["collection_id"] == str(test_collection.id)

    by_slug = client.get(
        "/v1/documents",
        params={"collection_slug": "test-collection"},
        headers=auth_headers,
    )
    assert by_slug.status_code == 200
    assert len(by_slug.json()) == 1

    by_id = client.get(
        "/v1/documents",
        params={"collection_id": str(test_collection.id)},
        headers=auth_headers,
    )
    assert by_id.status_code == 200
    assert len(by_id.json()) == 1

    fetched = client.get(f"/v1/documents/{document_id}", headers=auth_headers)
    assert fetched.status_code == 200
    assert fetched.json()["title"] == "hello.md"
    assert fetched.json()["content"] == "# Hello\n\nWorld."

    status = client.get(f"/v1/documents/{document_id}/status", headers=auth_headers)
    assert status.status_code == 200
    assert status.json()["document_id"] == document_id
    assert status.json()["status"] == "queued"

    patched = client.patch(
        f"/v1/documents/{document_id}",
        json={"title": "renamed.md"},
        headers=auth_headers,
    )
    assert patched.status_code == 200
    assert patched.json()["title"] == "renamed.md"
    assert mock_trigger.call_count == 2

    deleted = client.delete(f"/v1/documents/{document_id}", headers=auth_headers)
    assert deleted.status_code == 204
    assert deleted.content == b""

    missing = client.get(f"/v1/documents/{document_id}", headers=auth_headers)
    assert missing.status_code == 404
    assert missing.json()["error"] == "not_found"


@patch("app.services.documents.trigger_process_document")
def test_upload_multipart(
    mock_trigger, client: TestClient, auth_headers, test_collection
) -> None:
    response = client.post(
        "/v1/documents",
        files={"file": ("notes.md", b"# Notes\n", "text/markdown")},
        data={"collection_slug": "test-collection"},
        headers=auth_headers,
    )
    assert response.status_code == 202
    body = response.json()
    assert body["filename"] == "notes.md"
    assert body["accepted"] is True
    mock_trigger.assert_called_once_with(body["document_id"])


@patch("app.services.documents.trigger_process_document")
def test_upload_json_by_collection_id(
    mock_trigger, client: TestClient, auth_headers, test_collection
) -> None:
    response = client.post(
        "/v1/documents/json",
        json={
            "content": "# Hello",
            "filename": "by-id.md",
            "collection_id": str(test_collection.id),
        },
        headers=auth_headers,
    )
    assert response.status_code == 202
    assert response.json()["filename"] == "by-id.md"
    mock_trigger.assert_called_once()


@patch("app.services.documents.trigger_process_document")
def test_upload_duplicate_filename_returns_409(
    mock_trigger, client: TestClient, auth_headers, test_collection
) -> None:
    first = client.post("/v1/documents/json", json=JSON_BODY, headers=auth_headers)
    assert first.status_code == 202

    second = client.post("/v1/documents/json", json=JSON_BODY, headers=auth_headers)
    assert second.status_code == 409
    assert second.json()["error"] == "conflict"


def test_upload_rejects_non_markdown(
    client: TestClient, auth_headers, test_collection
) -> None:
    response = client.post(
        "/v1/documents/json",
        json={
            "content": "hello",
            "filename": "hello.txt",
            "collection_slug": "test-collection",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_upload_missing_collection_returns_404(
    client: TestClient, auth_headers
) -> None:
    response = client.post("/v1/documents/json", json=JSON_BODY, headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_upload_requires_one_collection_identifier(
    client: TestClient, auth_headers, test_collection
) -> None:
    both = client.post(
        "/v1/documents/json",
        json={
            "content": "# Hello",
            "filename": "hello.md",
            "collection_id": str(test_collection.id),
            "collection_slug": "test-collection",
        },
        headers=auth_headers,
    )
    assert both.status_code == 422
    assert both.json()["error"] == "validation_error"

    neither = client.post(
        "/v1/documents/json",
        json={"content": "# Hello", "filename": "hello.md"},
        headers=auth_headers,
    )
    assert neither.status_code == 422
    assert neither.json()["error"] == "validation_error"


def test_multipart_requires_one_collection_identifier(
    client: TestClient, auth_headers, test_collection
) -> None:
    response = client.post(
        "/v1/documents",
        files={"file": ("notes.md", b"# Notes\n", "text/markdown")},
        data={
            "collection_id": str(test_collection.id),
            "collection_slug": "test-collection",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_list_rejects_both_collection_filters(
    client: TestClient, auth_headers, test_collection
) -> None:
    response = client.get(
        "/v1/documents",
        params={
            "collection_id": str(test_collection.id),
            "collection_slug": "test-collection",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_list_unknown_collection_slug_returns_404(
    client: TestClient, auth_headers
) -> None:
    response = client.get(
        "/v1/documents",
        params={"collection_slug": "missing"},
        headers=auth_headers,
    )
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_get_document_not_found(client: TestClient, auth_headers) -> None:
    response = client.get(f"/v1/documents/{uuid4()}", headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_get_document_rejects_other_tenant(
    client: TestClient, auth_headers, test_collection, db_session
) -> None:
    with patch("app.services.documents.trigger_process_document"):
        uploaded = client.post(
            "/v1/documents/json", json=JSON_BODY, headers=auth_headers
        )
    assert uploaded.status_code == 202
    _, other_key = create_system_user(db_session, name="Other Tenant")
    response = client.get(
        f"/v1/documents/{uploaded.json()['document_id']}",
        headers={"Authorization": f"Bearer {other_key}"},
    )
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


@patch("app.services.documents.trigger_process_document")
def test_patch_invalid_content_returns_422(
    mock_trigger, client: TestClient, auth_headers, test_collection
) -> None:
    uploaded = client.post("/v1/documents/json", json=JSON_BODY, headers=auth_headers)
    assert uploaded.status_code == 202

    response = client.patch(
        f"/v1/documents/{uploaded.json()['document_id']}",
        json={"content": "   \n"},
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_patch_empty_body_returns_422(
    client: TestClient, auth_headers, test_collection
) -> None:
    with patch("app.services.documents.trigger_process_document"):
        uploaded = client.post(
            "/v1/documents/json", json=JSON_BODY, headers=auth_headers
        )
    response = client.patch(
        f"/v1/documents/{uploaded.json()['document_id']}",
        json={},
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
