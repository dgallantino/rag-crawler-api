"""Tests for markdown chunking."""

from unittest.mock import Mock

import pytest

from app.rag.processor import MarkdownProcessor


class _FakeEncoding:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


@pytest.fixture
def processor() -> MarkdownProcessor:
    return MarkdownProcessor(
        Mock(),
        "test-model",
        chunk_max_tokens=500,
        chunk_min_tokens=1,
        chunk_overlap_percent=10,
    )


@pytest.fixture(autouse=True)
def mock_tiktoken(monkeypatch):
    monkeypatch.setattr("app.rag.processor._get_encoding", lambda: _FakeEncoding())


def test_chunk_extracts_section_header_and_all_headings(processor: MarkdownProcessor) -> None:
    text = (
        "# Intro\n\nShort intro.\n\n"
        "## Details\n\n"
        + ("Second paragraph with enough words to form its own chunk. " * 40) +
        "\n\n### Subsection\n\n"
        + ("Subsection text that forces another chunk. " * 40)
    )
    chunks = processor.chunk(text)

    assert len(chunks) >= 2
    all_chunk_headings = [chunk.metadata.get("all_headings", []) for chunk in chunks]
    assert any(headings == ["Intro"] for headings in all_chunk_headings)
    assert any(headings == ["Details"] for headings in all_chunk_headings)
    assert any(headings == ["Subsection"] for headings in all_chunk_headings)
    assert not any(len(headings) == 3 for headings in all_chunk_headings)

    assert any(chunk.metadata.get("section_header") == "Details" for chunk in chunks)
    assert any(chunk.metadata.get("section_header") == "Subsection" for chunk in chunks)


def test_chunk_extracts_frontmatter_tags(processor: MarkdownProcessor) -> None:
    text = "---\ntags: [python, rag]\n---\n\n# Body\n\nContent here."
    chunks = processor.chunk(text)

    assert chunks
    assert chunks[0].metadata.get("tags") == ["python", "rag"]


def test_chunk_merges_undersized_chunks() -> None:
    from app.rag.processor import _token_length

    processor = MarkdownProcessor(
        Mock(),
        "test-model",
        chunk_max_tokens=120,
        chunk_min_tokens=60,
        chunk_overlap_percent=0,
    )
    # Three short paragraphs that each fall under min when split on blank lines.
    text = "aaaa " * 10 + "\n\n" + "bbbb " * 10 + "\n\n" + "cccc " * 10
    chunks = processor.chunk(text)

    assert chunks
    assert all(_token_length(c.content) <= 120 for c in chunks)
    for chunk in chunks[:-1]:
        assert _token_length(chunk.content) >= 60
    assert len(chunks) < 3


def test_chunk_leaves_short_tail_when_merge_exceeds_max() -> None:
    from app.rag.processor import _token_length

    processor = MarkdownProcessor(
        Mock(),
        "test-model",
        chunk_max_tokens=50,
        chunk_min_tokens=40,
        chunk_overlap_percent=0,
    )
    # First chunk near max; second undersized but cannot merge without exceeding max.
    text = ("x" * 48) + "\n\n" + ("y" * 20)
    chunks = processor.chunk(text)

    assert len(chunks) == 2
    assert _token_length(chunks[0].content) <= 50
    assert _token_length(chunks[1].content) < 40
    assert all(_token_length(c.content) <= 50 for c in chunks)