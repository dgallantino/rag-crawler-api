from __future__ import annotations

from httpx import get
from pytest import Session

from app.config import Settings, get_settings
from app.rag.generation import answer_with_retrieval, normalize_chunks
from app.rag.rerank import rerank
from app.rag.processor import MarkdownProcessor
from app.rag.retrieval import EmbedFn, retrieve

from openai import OpenAI

settings = get_settings()

def rag(
    user_id: str,
    query: str,
    top_k: int,
    filters: dict | None,
    collection: str | None,
    *,
    session: Session,
) -> dict:
    """Retrieve relevant chunks for a query.

    This is a black-box stub. The real implementation (embedding, vector
    search, reranking) is out of scope and will be provided separately.
    """
    settings = get_settings()
    embed_fn = create_embed_fn(settings)
    initial_retrieve_k = top_k 
    candidates = retrieve(
        query, initial_retrieve_k, filters, collection,
        session=session, embed_fn=embed_fn)
    
    completion_client = create_openai_client(settings)
    answer = answer_with_retrieval(
        query,
        normalize_chunks(candidates),
        completion_client,
        completion_model=settings.completion_model,
    )
    return answer


def create_openai_client(settings: Settings) -> OpenAI:
    """Embedding client factory."""
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    return OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )


def create_embed_fn(settings: Settings) -> EmbedFn:
    """Return a single-text embedder for query retrieval."""
    client = create_openai_client(settings)
    model = settings.embedding_model

    def embed(text: str) -> list[float]:
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding

    return embed


def create_markdown_processor(settings: Settings) -> MarkdownProcessor:
    """Create a RAG processor from application settings."""
    return MarkdownProcessor(
        create_openai_client(settings),
        settings.embedding_model,
        chunk_max_tokens=settings.chunk_max_tokens,
        chunk_overlap_percent=settings.chunk_overlap_percent,
    )


