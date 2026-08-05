"""Tests for app.services.rag — retrieval orchestration and client factories."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.rag.chunks import ScoredChunk
from app.rag.generation import RagResponse
from app.rag.retrieval import Retriever
from app.services.rag import (
    answer_service,
    create_embed_fn,
    create_openai_client,
    create_rerank_fn,
    get_retriever,
    retrieval_service,
)


def _mock_rag_settings(monkeypatch) -> MagicMock:
    settings = MagicMock()
    settings.openrouter_api_key = "sk-test"
    settings.openrouter_base_url = "https://openrouter.ai/api/v1"
    settings.embedding_model = "openai/text-embedding-3-small"
    settings.completion_model = "test-completion-model"
    settings.rerank_model = "cohere/rerank-v3.5"
    settings.rerank_expansion_factor = 4
    monkeypatch.setattr("app.services.rag.get_settings", lambda: settings)
    get_retriever.cache_clear()
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
        ScoredChunk(chunk=chunk_a, score=0.55, retrieval_score=0.55),
        ScoredChunk(chunk=chunk_b, score=0.66, retrieval_score=0.66),
        ScoredChunk(chunk=chunk_c, score=0.77, retrieval_score=0.77),
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
    assert result[0].score == 0.99
    assert result[0].rerank_score == 0.99
    assert result[0].retrieval_score == 0.77
    assert result[1].chunk is chunk_a
    assert result[1].score == 0.42
    assert result[1].rerank_score == 0.42
    assert result[1].retrieval_score == 0.55


def test_create_rerank_fn_raises_on_http_error(monkeypatch) -> None:
    import httpx

    def fake_post(*args, **kwargs):
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/rerank")
        return httpx.Response(502, request=request)

    monkeypatch.setattr("app.services.rag.httpx.post", fake_post)

    chunk = MagicMock()
    chunk.content = "Some content"
    chunks = [ScoredChunk(chunk=chunk, score=0.9, retrieval_score=0.9)]

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
    retrieved = [ScoredChunk(chunk=chunk, score=0.9, retrieval_score=0.9)]

    fake = MagicMock()
    fake.retrieve.return_value = retrieved
    monkeypatch.setattr("app.services.rag.get_retriever", lambda: fake)

    result = retrieval_service(
        query="What is the SLA?",
        top_k=3,
        filters={"metadata": {"doc_type": "contract"}},
        user=user,
        collection_slug="test-collection",
        session=db_session,
    )

    assert result == retrieved
    fake.retrieve.assert_called_once_with(
        "What is the SLA?",
        3,
        {"metadata": {"doc_type": "contract"}},
        [str(test_collection.id)],
        session=db_session,
        use_rerank=False,
    )


def test_retrieval_service_rejects_other_users_collection_slug(
    db_session, test_user, monkeypatch
):
    from app.services.collections import CollectionNotFoundError, create_collection
    from app.services.system_user import create_system_user

    _mock_rag_settings(monkeypatch)
    user, _ = test_user
    other_user, _ = create_system_user(db_session, name="Other Tenant")
    create_collection(db_session, other_user, name="Other Docs", slug="other-docs")

    fake = MagicMock()
    monkeypatch.setattr("app.services.rag.get_retriever", lambda: fake)

    with pytest.raises(CollectionNotFoundError):
        retrieval_service(
            query="q",
            top_k=1,
            filters=None,
            user=user,
            collection_slug="other-docs",
            session=db_session,
        )

    fake.retrieve.assert_not_called()


def test_retrieval_service_slug_none_passes_all_user_collections(
    db_session, test_user, test_collection, monkeypatch
):
    from app.services.collections import create_collection

    _mock_rag_settings(monkeypatch)
    user, _ = test_user
    other = create_collection(
        db_session, user, name="Other Collection", slug="other-collection"
    )

    fake = MagicMock()
    fake.retrieve.return_value = []
    monkeypatch.setattr("app.services.rag.get_retriever", lambda: fake)

    retrieval_service(
        query="q",
        top_k=1,
        filters=None,
        user=user,
        collection_slug=None,
        session=db_session,
    )

    fake.retrieve.assert_called_once()
    args, kwargs = fake.retrieve.call_args
    assert set(args[3]) == {
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
    fake = MagicMock()
    monkeypatch.setattr("app.services.rag.get_retriever", lambda: fake)

    with pytest.raises(CollectionNotFoundError):
        retrieval_service(
            query="q",
            top_k=1,
            filters=None,
            user=user,
            collection_slug=None,
            session=db_session,
        )

    fake.retrieve.assert_not_called()


def test_answer_service_generates_answer(monkeypatch):
    _mock_rag_settings(monkeypatch)
    chunk = MagicMock()
    chunk.id = "chunk-1"
    candidates = [ScoredChunk(chunk=chunk, score=0.9, retrieval_score=0.9)]
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


def test_retrieval_service_passes_use_rerank_false_by_default(
    db_session, test_user, test_collection, monkeypatch
):
    _mock_rag_settings(monkeypatch)
    user, _ = test_user

    fake = MagicMock()
    fake.retrieve.return_value = []
    monkeypatch.setattr("app.services.rag.get_retriever", lambda: fake)

    retrieval_service(
        query="q",
        top_k=1,
        filters=None,
        user=user,
        collection_slug=None,
        session=db_session,
    )

    assert fake.retrieve.call_args.kwargs["use_rerank"] is False


def test_retrieval_service_passes_use_rerank_true(
    db_session, test_user, test_collection, monkeypatch
):
    _mock_rag_settings(monkeypatch)
    user, _ = test_user

    fake = MagicMock()
    fake.retrieve.return_value = []
    monkeypatch.setattr("app.services.rag.get_retriever", lambda: fake)

    retrieval_service(
        query="q",
        top_k=5,
        filters=None,
        user=user,
        collection_slug="test-collection",
        use_rerank=True,
        session=db_session,
    )

    fake.retrieve.assert_called_once_with(
        "q",
        5,
        None,
        [str(test_collection.id)],
        session=db_session,
        use_rerank=True,
    )


def test_retriever_overfetches_when_rerank_enabled(monkeypatch, db_session) -> None:
    """With use_rerank=True and top_k=5, _retrieve is called with top_k=20."""
    captured: dict = {}

    def mock_retrieve(query, top_k, filters, collection, *, session, embed_fn):
        captured["top_k"] = top_k
        return [
            ScoredChunk(chunk=MagicMock(), score=0.9, retrieval_score=0.9)
            for _ in range(top_k)
        ]

    monkeypatch.setattr("app.rag.retrieval._retrieve", mock_retrieve)

    rerank_calls: list[tuple] = []

    def mock_rerank_fn(query, top_k, candidates):
        rerank_calls.append((query, top_k, len(candidates)))
        return candidates[:top_k]

    retriever = Retriever(
        embed_fn=lambda text: [0.0] * 3,
        rerank_fn=mock_rerank_fn,
        rerank_expansion_factor=4,
    )
    results = retriever.retrieve(
        "q",
        5,
        None,
        ["collection-id"],
        session=db_session,
        use_rerank=True,
    )

    assert captured["top_k"] == 20
    assert len(rerank_calls) == 1
    assert rerank_calls[0] == ("q", 5, 20)
    assert len(results) == 5


def test_retriever_does_not_rerank_when_disabled(monkeypatch, db_session) -> None:
    captured: dict = {}

    def mock_retrieve(query, top_k, filters, collection, *, session, embed_fn):
        captured["top_k"] = top_k
        return [
            ScoredChunk(chunk=MagicMock(), score=0.9, retrieval_score=0.9)
            for _ in range(top_k)
        ]

    monkeypatch.setattr("app.rag.retrieval._retrieve", mock_retrieve)

    rerank_called = False

    def mock_rerank_fn(query, top_k, candidates):
        nonlocal rerank_called
        rerank_called = True
        return candidates[:top_k]

    retriever = Retriever(
        embed_fn=lambda text: [0.0] * 3,
        rerank_fn=mock_rerank_fn,
        rerank_expansion_factor=4,
    )
    results = retriever.retrieve(
        "q",
        3,
        None,
        ["collection-id"],
        session=db_session,
        use_rerank=False,
    )

    assert captured["top_k"] == 3
    assert rerank_called is False
    assert len(results) == 3


def test_retriever_falls_back_when_rerank_fn_missing(monkeypatch, db_session) -> None:
    candidates = [
        ScoredChunk(chunk=MagicMock(id=i), score=float(i), retrieval_score=float(i))
        for i in range(8)
    ]

    monkeypatch.setattr(
        "app.rag.retrieval._retrieve",
        lambda *a, **k: candidates,
    )

    retriever = Retriever(
        embed_fn=lambda text: [0.0] * 3,
        rerank_fn=None,
        rerank_expansion_factor=4,
    )
    results = retriever.retrieve(
        "q",
        3,
        None,
        ["collection-id"],
        session=db_session,
        use_rerank=True,
    )

    assert [r.score for r in results] == [0.0, 1.0, 2.0]
