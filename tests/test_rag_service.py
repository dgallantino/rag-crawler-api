"""Tests for app.services.rag — retrieval orchestration and client factories."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.rag.generation import RagResponse, ScoredChunk
from app.rag.retrieval import RetrievedChunk
from app.services.rag import create_embed_fn, create_openai_client, rag


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


def test_rag_retrieves_and_generates_answer(db_session, test_collection, monkeypatch):
    _mock_rag_settings(monkeypatch)
    chunk = MagicMock()
    chunk.id = "chunk-1"
    retrieved = [RetrievedChunk(chunk=chunk, similarity_score=0.9)]
    expected = RagResponse(answer="Generated answer", sources=[])

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

    def mock_normalize(chunks):
        return [ScoredChunk(chunk=chunk, score=0.9)]

    monkeypatch.setattr("app.services.rag.retrieve", mock_retrieve)
    monkeypatch.setattr("app.services.rag.normalize_chunks", mock_normalize)
    monkeypatch.setattr(
        "app.services.rag.create_embed_fn",
        lambda settings: lambda text: [0.0] * 1536,
    )
    monkeypatch.setattr(
        "app.services.rag.create_openai_client",
        lambda settings: MagicMock(),
    )
    monkeypatch.setattr(
        "app.services.rag.answer_with_retrieval",
        lambda query, chunks, client, *, completion_model: expected,
    )

    result = rag(
        user_id="user-1",
        query="What is the SLA?",
        top_k=3,
        filters={"metadata": {"doc_type": "contract"}},
        collection=str(test_collection.id),
        session=db_session,
    )

    assert result == expected
    assert retrieve_calls == [
        {
            "query": "What is the SLA?",
            "top_k": 3,
            "filters": {"metadata": {"doc_type": "contract"}},
            "collection": str(test_collection.id),
            "session": db_session,
        }
    ]


def test_rag_does_not_call_rerank(db_session, monkeypatch):
    """Rerank is not wired into the service yet."""
    _mock_rag_settings(monkeypatch)
    rerank_called = False

    def mock_rerank(*args, **kwargs):
        nonlocal rerank_called
        rerank_called = True
        return []

    monkeypatch.setattr("app.services.rag.rerank", mock_rerank)
    monkeypatch.setattr("app.services.rag.retrieve", lambda *a, **k: [])
    monkeypatch.setattr("app.services.rag.normalize_chunks", lambda chunks: [])
    monkeypatch.setattr(
        "app.services.rag.create_embed_fn",
        lambda settings: lambda text: [0.0] * 1536,
    )
    monkeypatch.setattr(
        "app.services.rag.create_openai_client",
        lambda settings: MagicMock(),
    )
    monkeypatch.setattr(
        "app.services.rag.answer_with_retrieval",
        lambda *a, **k: RagResponse(answer="", sources=[]),
    )

    rag(
        user_id="user-1",
        query="q",
        top_k=1,
        filters=None,
        collection=None,
        session=db_session,
    )

    assert rerank_called is False
