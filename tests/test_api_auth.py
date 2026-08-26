"""HTTP tests for API-key authentication on /v1 routes."""

from fastapi.testclient import TestClient

RETRIEVE_BODY = {"query": "what is this"}
DOCUMENT_JSON_BODY = {
    "content": "# Hello",
    "filename": "hello.md",
    "collection_slug": "test-collection",
}


def _assert_unauthorized(response) -> None:
    assert response.status_code == 401
    body = response.json()
    assert body["error"] == "unauthorized"
    assert body["message"] == "Invalid API key"
    assert body["request_id"]
    assert response.headers.get("www-authenticate") == "Bearer"


def test_health_unauthenticated(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_db_unauthenticated(client: TestClient) -> None:
    response = client.get("/health/db")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_collections_missing_header_returns_401(client: TestClient) -> None:
    _assert_unauthorized(client.get("/v1/collections"))


def test_collections_non_bearer_scheme_returns_401(client: TestClient) -> None:
    _assert_unauthorized(
        client.get("/v1/collections", headers={"Authorization": "Token not-a-bearer"})
    )


def test_collections_wrong_key_returns_401(client: TestClient) -> None:
    _assert_unauthorized(
        client.get("/v1/collections", headers={"Authorization": "Bearer wrong-key"})
    )


def test_collections_valid_key_lists_empty(client: TestClient, auth_headers) -> None:
    response = client.get("/v1/collections", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


def test_retrieve_missing_header_returns_401(client: TestClient) -> None:
    _assert_unauthorized(client.post("/v1/retrieve", json=RETRIEVE_BODY))


def test_retrieve_valid_key_without_collections_returns_404(
    client: TestClient, auth_headers
) -> None:
    response = client.post("/v1/retrieve", json=RETRIEVE_BODY, headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_documents_json_missing_header_returns_401(client: TestClient) -> None:
    _assert_unauthorized(client.post("/v1/documents/json", json=DOCUMENT_JSON_BODY))


def test_documents_json_valid_key_missing_collection_returns_404(
    client: TestClient, auth_headers
) -> None:
    response = client.post(
        "/v1/documents/json", json=DOCUMENT_JSON_BODY, headers=auth_headers
    )
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"
