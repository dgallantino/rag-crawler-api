"""Tests for embedding service client factory."""

import pytest

from app.config import Settings
from app.services.rag import create_openai_client


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
