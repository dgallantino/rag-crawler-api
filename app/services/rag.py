
from app.config import Settings
from app.rag.processor import MarkdownProcessor

from openai import OpenAI

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


