"""Tests for contextualized embedding text helper."""

from app.utils import build_contextualized_embedding_text


def test_neither_title_nor_heading_returns_body() -> None:
    assert build_contextualized_embedding_text("body only") == "body only"


def test_empty_strings_treated_as_absent() -> None:
    assert build_contextualized_embedding_text("body", title="", heading="") == "body"


def test_title_only() -> None:
    assert (
        build_contextualized_embedding_text("body", title="Doc Title")
        == "Title: Doc Title\n\nbody"
    )


def test_heading_only() -> None:
    assert (
        build_contextualized_embedding_text("body", heading="Section")
        == "Heading: Section\n\nbody"
    )


def test_title_and_heading() -> None:
    assert (
        build_contextualized_embedding_text(
            "body",
            title="Doc Title",
            heading="Section",
        )
        == "Title: Doc Title\nHeading: Section\n\nbody"
    )
