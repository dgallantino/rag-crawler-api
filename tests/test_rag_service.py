"""Tests for app.services.rag — retrieval orchestration and client factories."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.rag.generation import RagResponse, ScoredChunk
from app.rag.retrieval import RetrievedChunk
from app.services.rag import (
        answer_service,
        create_embed_fn,
        create_openai_client,
        create_rerank_fn,
        retrieval_service,
    )


def _mock_rag_settings(monkeypatch) -> MagicMock:
    settings = MagicMock()
    settings.openrouter_api_key = "sk-test"
    settings.openrouter_base_url = "https://openrouter.ai/api/v1"
    settings.embedding_model = "openai/text-embedding-3-small"
    settings.completion_model = "test-completion-model"
    monkeypatch.setattr("app.services.rag.get_settings", lambda: settings)
    return settings


def test_create_openai_client_raises_when_api_key_missing() -> None:
    settings = Settings(
        api_key_hash_secret="test-secret",
        openrouter_api_key="",
    )
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not configured"):
        create_openai_client(settings)


def test_create_openai_client_uses_settings(monkeypatch) -> None:
    captured: dict = {}

    class FakeOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    monkeypatch.setattr("app.services.rag.OpenAI", FakeOpenAI)

    settings = Settings(
        api_key_hash_secret="test-secret",
        openrouter_api_key="sk-test",
        openrouter_base_url="https://custom.example/v1",
    )
    create_openai_client(settings)

    assert captured == {
        "base_url": "https://custom.example/v1",
        "api_key": "sk-test",
    }


def test_create_embed_fn_embeds_text(monkeypatch) -> None:
    captured: dict = {}

    class FakeEmbeddings:
        def create(self, *, model: str, input: str) -> object:
            captured["model"] = model
            captured["input"] = input
            return type(
                "Response",
                (),
                {"data": [type("Item", (), {"embedding": [0.1, 0.2, 0.3]})()]},
            )()

    class FakeOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr("app.services.rag.OpenAI", FakeOpenAI)

    settings = Settings(
        api_key_hash_secret="test-secret",
        openrouter_api_key="sk-test",
        embedding_model="openai/text-embedding-3-small",
    )
    embed = create_embed_fn(settings)

    assert embed("hello world") == [0.1, 0.2, 0.3]
    assert captured == {
        "model": "openai/text-embedding-3-small",
        "input": "hello world",
    }


def _rerank_settings(**overrides) -> Settings:
    defaults = {
        "api_key_hash_secret": "test-secret",
        "openrouter_api_key": "sk-test",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_create_rerank_fn_raises_when_api_key_missing() -> None:
    settings = Settings(
        api_key_hash_secret="test-secret",
        openrouter_api_key="",
    )
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not configured"):
        create_rerank_fn(settings)


def test_create_rerank_fn_remapped_chunks(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "index": 2,
                        "relevance_score": 0.99,
                        "document": {"text": "SLA is 24 hours"},
                    },
                    {
                        "index": 0,
                        "relevance_score": 0.42,
                        "document": {"text": "Pricing tiers"},
                    },
                ]
            }

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("app.services.rag.httpx.post", fake_post)

    chunk_a = MagicMock()
    chunk_a.content = "Pricing tiers"
    chunk_b = MagicMock()
    chunk_b.content = "Support contacts"
    chunk_c = MagicMock()
    chunk_c.content = "SLA is 24 hours"

    chunks = [
        RetrievedChunk(chunk=chunk_a, similarity_score=0.55),
        RetrievedChunk(chunk=chunk_b, similarity_score=0.66),
        RetrievedChunk(chunk=chunk_c, similarity_score=0.77),
    ]

    rerank = create_rerank_fn(_rerank_settings())
    result = rerank("What is the SLA?", 2, chunks)

    assert captured == {
        "url": "https://openrouter.ai/api/v1/rerank",
        "headers": {
            "Authorization": "Bearer sk-test",
            "Content-Type": "application/json",
        },
        "json": {
            "model": "cohere/rerank-v3.5",
            "query": "What is the SLA?",
            "documents": [
                "Pricing tiers",
                "Support contacts",
                "SLA is 24 hours",
            ],
            "top_n": 2,
        },
        "timeout": 30.0,
    }
    assert len(result) == 2
    assert result[0].chunk is chunk_c
    assert result[0].rerank_score == 0.99
    assert result[0].similarity_score == 0.77
    assert result[1].chunk is chunk_a
    assert result[1].rerank_score == 0.42
    assert result[1].similarity_score == 0.55


def test_create_rerank_fn_raises_on_http_error(monkeypatch) -> None:
    import httpx

    def fake_post(*args, **kwargs):
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/rerank")
        return httpx.Response(502, request=request)

    monkeypatch.setattr("app.services.rag.httpx.post", fake_post)

    chunk = MagicMock()
    chunk.content = "Some content"
    chunks = [RetrievedChunk(chunk=chunk, similarity_score=0.9)]

    rerank = create_rerank_fn(_rerank_settings())

    with pytest.raises(httpx.HTTPStatusError):
        rerank("query", 1, chunks)


def test_create_rerank_fn_returns_empty_for_no_chunks() -> None:
    rerank = create_rerank_fn(_rerank_settings())
    assert rerank("query", 3, []) == []


def test_retrieval_service_retrieves_chunks(
    db_session, test_user, test_collection, monkeypatch
):
    _mock_rag_settings(monkeypatch)
    user, _ = test_user
    chunk = MagicMock()
    chunk.id = "chunk-1"
    retrieved = [RetrievedChunk(chunk=chunk, similarity_score=0.9)]

    retrieve_calls: list[dict] = []

    def mock_retrieve(query, top_k, filters, collection, *, session, embed_fn):
        retrieve_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "filters": filters,
                "collection": collection,
                "session": session,
            }
        )
        return retrieved

    monkeypatch.setattr("app.services.rag.retrieve", mock_retrieve)
    monkeypatch.setattr(
        "app.services.rag.create_embed_fn",
        lambda settings: lambda text: [0.0] * 1536,
    )

    result = retrieval_service(
        query="What is the SLA?",
        top_k=3,
        filters={"metadata": {"doc_type": "contract"}},
        user=user,
        collection_slug="test-collection",
        session=db_session,
    )

    assert result == retrieved
    assert retrieve_calls == [
        {
            "query": "What is the SLA?",
            "top_k": 3,
            "filters": {"metadata": {"doc_type": "contract"}},
            "collection": [str(test_collection.id)],
            "session": db_session,
        }
    ]


def test_retrieval_service_rejects_other_users_collection_slug(
    db_session, test_user, monkeypatch
):
    from app.services.collections import CollectionNotFoundError, create_collection
    from app.services.system_user import create_system_user

    _mock_rag_settings(monkeypatch)
    user, _ = test_user
    other_user, _ = create_system_user(db_session, name="Other Tenant")
    create_collection(db_session, other_user, name="Other Docs", slug="other-docs")

    monkeypatch.setattr(
        "app.services.rag.create_embed_fn",
        lambda settings: lambda text: [0.0] * 1536,
    )

    with pytest.raises(CollectionNotFoundError):
        retrieval_service(
            query="q",
            top_k=1,
            filters=None,
            user=user,
            collection_slug="other-docs",
            session=db_session,
        )


def test_retrieval_service_slug_none_passes_all_user_collections(
    db_session, test_user, test_collection, monkeypatch
):
    from app.services.collections import create_collection

    _mock_rag_settings(monkeypatch)
    user, _ = test_user
    other = create_collection(
        db_session, user, name="Other Collection", slug="other-collection"
    )
    retrieve_calls: list[dict] = []

    def mock_retrieve(query, top_k, filters, collection, *, session, embed_fn):
        retrieve_calls.append({"collection": collection})
        return []

    monkeypatch.setattr("app.services.rag.retrieve", mock_retrieve)
    monkeypatch.setattr(
        "app.services.rag.create_embed_fn",
        lambda settings: lambda text: [0.0] * 1536,
    )

    retrieval_service(
        query="q",
        top_k=1,
        filters=None,
        user=user,
        collection_slug=None,
        session=db_session,
    )

    assert len(retrieve_calls) == 1
    assert set(retrieve_calls[0]["collection"]) == {
        str(test_collection.id),
        str(other.id),
    }


def test_retrieval_service_slug_none_raises_when_user_has_no_collections(
    db_session, monkeypatch
):
    from app.services.collections import CollectionNotFoundError
    from app.services.system_user import create_system_user

    _mock_rag_settings(monkeypatch)
    user, _ = create_system_user(db_session, name="Empty Tenant")
    monkeypatch.setattr(
        "app.services.rag.create_embed_fn",
        lambda settings: lambda text: [0.0] * 1536,
    )

    with pytest.raises(CollectionNotFoundError):
        retrieval_service(
            query="q",
            top_k=1,
            filters=None,
            user=user,
            collection_slug=None,
            session=db_session,
        )


def test_answer_service_generates_answer(monkeypatch):
    _mock_rag_settings(monkeypatch)
    chunk = MagicMock()
    chunk.id = "chunk-1"
    candidates = [RetrievedChunk(chunk=chunk, similarity_score=0.9)]
    expected = RagResponse(answer="Generated answer", sources=[])
    captured: dict = {}

    monkeypatch.setattr(
        "app.services.rag.create_openai_client",
        lambda settings: MagicMock(),
    )

    def mock_answer(query, chunks, client, *, completion_model, max_tokens_context=None):
        captured["max_tokens_context"] = max_tokens_context
        return expected

    monkeypatch.setattr("app.services.rag.answer_with_retrieval", mock_answer)

    result = answer_service(
        query="What is the SLA?",
        candidates=candidates,
        max_tokens_context=128,
    )

    assert result == expected
    assert captured["max_tokens_context"] == 128


def test_retrieval_service_does_not_call_rerank_by_default(
    db_session, test_user, test_collection, monkeypatch
):
    _mock_rag_settings(monkeypatch)
    user, _ = test_user
    rerank_called = False

    def mock_rerank(*args, **kwargs):
        nonlocal rerank_called
        rerank_called = True
        return []

    monkeypatch.setattr("app.services.rag.rerank", mock_rerank)
    monkeypatch.setattr("app.services.rag.retrieve", lambda *a, **k: [])
    monkeypatch.setattr(
        "app.services.rag.create_embed_fn",
        lambda settings: lambda text: [0.0] * 1536,
    )

    retrieval_service(
        query="q",
        top_k=1,
        filters=None,
        user=user,
        collection_slug=None,
        session=db_session,
    )

    assert rerank_called is False
