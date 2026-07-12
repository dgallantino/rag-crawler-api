"""Unit tests for POST /v1/query with mocked OpenRouter."""

from __future__ import annotations

from uuid import uuid4

import pytest


def _bearer_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


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


def test_query_unknown_user_returns_404(client, test_settings) -> None:
    response = client.post(
        "/v1/query",
        json={
            "user_id": str(uuid4()),
            "query": "What is the SLA?",
        },
        headers=_bearer_headers(test_settings.internal_bearer_token),
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"


def test_query_unknown_collection_returns_404(
    client, test_user, test_settings
) -> None:
    user, _ = test_user
    response = client.post(
        "/v1/query",
        json={
            "user_id": str(user.id),
            "query": "What is the SLA?",
            "collection": "missing-collection",
        },
        headers=_bearer_headers(test_settings.internal_bearer_token),
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Collection not found"


def test_query_returns_matching_chunks(
    client,
    db_session,
    test_user,
    test_collection,
    test_settings,
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
        headers=_bearer_headers(test_settings.internal_bearer_token),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_used"] == "What is the enterprise SLA?"
    assert payload["reranked"] is False
    assert len(payload["results"]) == 1
    assert payload["results"][0]["chunk_id"] == str(chunk.id)
    assert "99.9%" in payload["results"][0]["text"]
    assert payload["results"][0]["score"] == pytest.approx(1.0, abs=1e-4)
