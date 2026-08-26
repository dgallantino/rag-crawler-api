"""HTTP tests for POST /v1/retrieve and POST /v1/query."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.rag.generation import RagResponse


@patch("app.api.query.retrieval_service")
def test_retrieve_forwards_fields(
    mock_retrieve, client: TestClient, auth_headers, test_collection
) -> None:
    mock_retrieve.return_value = []
    response = client.post(
        "/v1/retrieve",
        json={
            "query": "what is this",
            "top_k": 7,
            "use_rerank": True,
            "collection_slug": "test-collection",
            "filters": {"metadata": {"doc_type": "contract"}},
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query_used"] == "what is this"
    assert body["top_k"] == 7
    assert body["reranked"] is True
    assert body["results"] == []
    assert isinstance(body["latency_ms"], int)

    mock_retrieve.assert_called_once()
    kwargs = mock_retrieve.call_args.kwargs
    assert kwargs["query"] == "what is this"
    assert kwargs["top_k"] == 7
    assert kwargs["use_rerank"] is True
    assert kwargs["collection_slug"] == "test-collection"
    assert kwargs["filters"]["metadata"] == {"doc_type": "contract"}
    assert kwargs["user"].name == "Test Tenant"
    assert kwargs["session"] is not None


@patch("app.api.query.answer_service")
@patch("app.api.query.retrieval_service")
def test_query_forwards_fields(
    mock_retrieve, mock_answer, client: TestClient, auth_headers, test_collection
) -> None:
    candidates = ["chunk-a"]
    mock_retrieve.return_value = candidates
    mock_answer.return_value = RagResponse(answer="grounded answer", sources=[])

    response = client.post(
        "/v1/query",
        json={
            "query": "what is this",
            "top_k": 3,
            "use_rerank": False,
            "collection_slug": "test-collection",
            "max_tokens_context": 256,
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json() == {"answer": "grounded answer", "sources": []}

    mock_retrieve.assert_called_once()
    retrieve_kwargs = mock_retrieve.call_args.kwargs
    assert retrieve_kwargs["query"] == "what is this"
    assert retrieve_kwargs["top_k"] == 3
    assert retrieve_kwargs["use_rerank"] is False
    assert retrieve_kwargs["collection_slug"] == "test-collection"
    assert retrieve_kwargs["filters"] is None

    mock_answer.assert_called_once_with(
        "what is this",
        candidates,
        max_tokens_context=256,
    )


def test_retrieve_missing_collection_returns_404(
    client: TestClient, auth_headers
) -> None:
    response = client.post(
        "/v1/retrieve",
        json={"query": "what is this"},
        headers=auth_headers,
    )
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_retrieve_unknown_slug_returns_404(
    client: TestClient, auth_headers, test_collection
) -> None:
    response = client.post(
        "/v1/retrieve",
        json={"query": "what is this", "collection_slug": "missing"},
        headers=auth_headers,
    )
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_query_missing_collection_returns_404(
    client: TestClient, auth_headers
) -> None:
    response = client.post(
        "/v1/query",
        json={"query": "what is this"},
        headers=auth_headers,
    )
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_retrieve_rejects_extra_fields(client: TestClient, auth_headers) -> None:
    response = client.post(
        "/v1/retrieve",
        json={"query": "what is this", "user_id": "nope"},
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
