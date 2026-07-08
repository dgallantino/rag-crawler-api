"""Tests for markdown chunking."""

import pytest

from app.rag.chunking import chunk_markdown


class _FakeEncoding:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


@pytest.fixture(autouse=True)
def mock_tiktoken(monkeypatch):
    monkeypatch.setattr("app.rag.chunking._get_encoding", lambda: _FakeEncoding())


def test_chunk_markdown_extracts_section_header_and_all_headings() -> None:
    text = (
        "# Intro\n\nShort intro.\n\n"
        "## Details\n\n"
        + ("Second paragraph with enough words to form its own chunk. " * 40) +
        "\n\n### Subsection\n\n"
        + ("Subsection text that forces another chunk. " * 40)
    )
    chunks = chunk_markdown(text, chunk_max_tokens=500, chunk_overlap_percent=10)

    assert len(chunks) >= 2
    headings_in_text = ["Intro", "Details", "Subsection"]

    # Check all_headings is present and matches for every chunk
    counts = [chunk.metadata.get("all_headings", []) for chunk in chunks]
    for headings in counts:
        assert len(headings) == len(headings_in_text)
        assert headings == headings_in_text

    # Also still test if section_header is set for one of the chunks
    assert any(chunk.metadata.get("section_header") == "Details" for chunk in chunks)
    assert any(chunk.metadata.get("section_header") == "Subsection" for chunk in chunks)


def test_chunk_markdown_extracts_frontmatter_tags() -> None:
    text = "---\ntags: [python, rag]\n---\n\n# Body\n\nContent here."
    chunks = chunk_markdown(text, chunk_max_tokens=500, chunk_overlap_percent=10)


    assert chunks
    assert chunks[0].metadata.get("tags") == ["python", "rag"]
