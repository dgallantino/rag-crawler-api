"""Unit tests for POST /v1/query with mocked OpenRouter."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.config import Settings


def _bearer_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def query_api_settings(monkeypatch) -> Settings:
    settings = Settings(
        api_key_hash_secret="test-secret",
        internal_bearer_token="test-bearer-token",
    )
    monkeypatch.setattr("app.dependencies.bearer.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.query.get_settings", lambda: settings)
    return settings


def test_query_requires_bearer_token(client, test_user, test_collection) -> None:
    user, _ = test_user
    response = client.post(
        "/v1/query",
        json={
            "user_id": str(user.id),
            "query": "What is the SLA?",
            "collection": test_collection.slug,
        },
    )
    assert response.status_code == 401


def test_query_unknown_user_returns_404(client, query_api_settings) -> None:
    response = client.post(
        "/v1/query",
        json={
            "user_id": str(uuid4()),
            "query": "What is the SLA?",
        },
        headers=_bearer_headers(query_api_settings.internal_bearer_token),
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"


def test_query_unknown_collection_returns_404(
    client, test_user, query_api_settings
) -> None:
    user, _ = test_user
    response = client.post(
        "/v1/query",
        json={
            "user_id": str(user.id),
            "query": "What is the SLA?",
            "collection": "missing-collection",
        },
        headers=_bearer_headers(query_api_settings.internal_bearer_token),
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Collection not found"


def test_query_returns_matching_chunks(
    client,
    db_session,
    test_user,
    test_collection,
    query_api_settings,
    monkeypatch,
) -> None:
    from tests.test_rag_retrieval import _make_chunk, _vec

    user, _ = test_user
    chunk = _make_chunk(
        db_session,
        test_collection,
        content="Enterprise SLA is 99.9% uptime",
        chunk_vector=_vec(1.0),
    )

    monkeypatch.setattr(
        "app.api.query.create_embed_fn",
        lambda settings: (lambda _text: _vec(1.0)),
    )

    response = client.post(
        "/v1/query",
        json={
            "user_id": str(user.id),
            "query": "What is the enterprise SLA?",
            "top_k": 3,
            "rerank": False,
            "collection": test_collection.slug,
        },
        headers=_bearer_headers(query_api_settings.internal_bearer_token),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_used"] == "What is the enterprise SLA?"
    assert payload["reranked"] is False
    assert len(payload["results"]) == 1
    assert payload["results"][0]["chunk_id"] == str(chunk.id)
    assert "99.9%" in payload["results"][0]["text"]
    assert payload["results"][0]["score"] == pytest.approx(1.0, abs=1e-4)
