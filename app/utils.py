"""Planned: Shared utility helpers used across the application."""

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_url(url: str) -> str:
    """Placeholder: normalize and validate URLs before crawling."""
    return url.strip().rstrip("/")

# anti pattern this utility is shared in crawler and rag IMPLEMENTATION
def build_contextualized_embedding_text(
    body: str,
    *,
    title: str | None = None,
    heading: str | None = None,
) -> str:
    """Prefix optional title/heading into text for the embedding model.

    Stored chunk content should remain ``body``; only the embed input is
    contextualized. Empty strings are treated as absent.
    """
    context_lines: list[str] = []
    if title:
        context_lines.append(f"Title: {title}")
    if heading:
        context_lines.append(f"Heading: {heading}")

    if not context_lines:
        return body

    return "\n".join(context_lines) + f"\n\n{body}"
