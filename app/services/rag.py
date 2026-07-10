from __future__ import annotations

from app.config import Settings
from app.rag.processor import MarkdownProcessor

from openai import OpenAI


def retrieve(
    user_id: str,
    query: str,
    top_k: int,
    filters: dict | None,
    rerank: bool,
    collection: str | None,
) -> dict:
    """Retrieve relevant chunks for a query.

    This is a black-box stub. The real implementation (embedding, vector
    search, reranking) is out of scope and will be provided separately.
    """
    raise NotImplementedError


def create_openai_client(settings: Settings) -> OpenAI:
    """Embedding client factory."""
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    return OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )


def create_markdown_processor(settings: Settings) -> MarkdownProcessor:
    """Create a RAG processor from application settings."""
    return MarkdownProcessor(
        create_openai_client(settings),
        settings.embedding_model,
        chunk_max_tokens=settings.chunk_max_tokens,
        chunk_overlap_percent=settings.chunk_overlap_percent,
    )


