"""HTTP tests for /v1/collections."""

from uuid import uuid4

from fastapi.testclient import TestClient

from app.services.system_user import create_system_user


def test_create_list_get_patch_delete_collection(
    client: TestClient, auth_headers
) -> None:
    created = client.post(
        "/v1/collections",
        json={"name": "Docs", "slug": "docs"},
        headers=auth_headers,
    )
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "Docs"
    assert body["slug"] == "docs"
    collection_id = body["id"]
    assert body["created_at"]

    listed = client.get("/v1/collections", headers=auth_headers)
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["id"] == collection_id

    fetched = client.get(f"/v1/collections/{collection_id}", headers=auth_headers)
    assert fetched.status_code == 200
    assert fetched.json()["slug"] == "docs"

    patched = client.patch(
        f"/v1/collections/{collection_id}",
        json={"name": "Renamed", "slug": "renamed"},
        headers=auth_headers,
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "Renamed"
    assert patched.json()["slug"] == "renamed"

    deleted = client.delete(f"/v1/collections/{collection_id}", headers=auth_headers)
    assert deleted.status_code == 204
    assert deleted.content == b""

    missing = client.get(f"/v1/collections/{collection_id}", headers=auth_headers)
    assert missing.status_code == 404
    assert missing.json()["error"] == "not_found"


def test_get_collection_not_found(client: TestClient, auth_headers) -> None:
    response = client.get(f"/v1/collections/{uuid4()}", headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"
    assert response.json()["request_id"]


def test_get_collection_rejects_other_tenant(
    client: TestClient, auth_headers, test_collection, db_session
) -> None:
    _, other_key = create_system_user(db_session, name="Other Tenant")
    response = client.get(
        f"/v1/collections/{test_collection.id}",
        headers={"Authorization": f"Bearer {other_key}"},
    )
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_create_duplicate_slug_returns_409(
    client: TestClient, auth_headers, test_collection
) -> None:
    response = client.post(
        "/v1/collections",
        json={"name": "Taken", "slug": test_collection.slug},
        headers=auth_headers,
    )
    assert response.status_code == 409
    assert response.json()["error"] == "conflict"


def test_patch_duplicate_slug_returns_409(
    client: TestClient, auth_headers, test_collection
) -> None:
    other = client.post(
        "/v1/collections",
        json={"name": "Second", "slug": "second"},
        headers=auth_headers,
    )
    assert other.status_code == 201

    response = client.patch(
        f"/v1/collections/{other.json()['id']}",
        json={"slug": test_collection.slug},
        headers=auth_headers,
    )
    assert response.status_code == 409
    assert response.json()["error"] == "conflict"


def test_patch_empty_body_returns_422(client: TestClient, auth_headers, test_collection) -> None:
    response = client.patch(
        f"/v1/collections/{test_collection.id}",
        json={},
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_create_rejects_extra_fields(client: TestClient, auth_headers) -> None:
    response = client.post(
        "/v1/collections",
        json={"name": "Docs", "unknown": True},
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
