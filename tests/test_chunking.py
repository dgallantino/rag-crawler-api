"""Tests for markdown chunking."""

import pytest

from app.rag.chunking import chunk_markdown


class _FakeEncoding:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


@pytest.fixture(autouse=True)
def mock_tiktoken(monkeypatch):
    monkeypatch.setattr("app.rag.chunking._get_encoding", lambda: _FakeEncoding())


def test_chunk_markdown_extracts_section_header() -> None:
    text = (
        "# Intro\n\nShort intro.\n\n## Details\n\n"
        + ("Second paragraph with enough words to form its own chunk. " * 80)
    )
    chunks = chunk_markdown(text)

    assert len(chunks) >= 2
    assert any(chunk.metadata.get("section_header") == "Details" for chunk in chunks)


def test_chunk_markdown_extracts_frontmatter_tags() -> None:
    text = "---\ntags: [python, rag]\n---\n\n# Body\n\nContent here."
    chunks = chunk_markdown(text)

    assert chunks
    assert chunks[0].metadata.get("tags") == ["python", "rag"]
