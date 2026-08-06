from __future__ import annotations

from functools import lru_cache

import httpx
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import SystemUser
from app.rag.chunks import ScoredChunk
from app.rag.generation import RagResponse, answer_with_retrieval
from app.rag.processor import MarkdownProcessor
from app.rag.retrieval import EmbedFn, RerankServiceFn, Retriever
from app.schemas.query import ChunkSource, RetrievalChunk, RetrievalResult
from app.services.collections import get_collection_by_slug

from openai import OpenAI


def retrieval_service(
    query: str,
    top_k: int,
    filters: dict | None,
    *,
    user: SystemUser,
    collection_slug: str | None = None,
    use_rerank: bool = False,
    session: Session,
) -> list[ScoredChunk]:
    """Retrieve and optionally rerank relevant chunks for a query.

    Resolves collections via ``get_collection_by_slug`` (raises
    ``CollectionNotFoundError`` when none match). Tenant scoping is entirely
    in that lookup — the retriever only receives collection UUID(s).
    """
    collections = get_collection_by_slug(session, user, collection_slug)
    collection_ids = [str(c.id) for c in collections]

    return get_retriever().retrieve(
        query,
        top_k,
        filters,
        collection_ids,
        session=session,
        use_rerank=use_rerank,
    )


def _chunk_to_retrieval_chunk(item: ScoredChunk) -> RetrievalChunk:
    chunk = item.chunk
    meta = chunk.chunk_metadata or {}
    source = None
    if chunk.document is not None:
        source = ChunkSource(
            document=chunk.document.title,
            page=meta.get("page"),
            url=chunk.document.url,
        )
    return RetrievalChunk(
        chunk_id=str(chunk.id),
        text=chunk.content,
        score=item.score,
        source=source,
    )


def chunks_to_retrieval_result(
    query: str,
    chunks: list[ScoredChunk],
    *,
    top_k: int,
    use_rerank: bool,
    latency_ms: int | None = None,
) -> RetrievalResult:
    """Shape retrieved chunks into the API RetrievalResult schema."""
    return RetrievalResult(
        results=[_chunk_to_retrieval_chunk(chunk) for chunk in chunks],
        query_used=query,
        latency_ms=latency_ms,
        top_k=top_k,
        reranked=use_rerank,
    )


def answer_service(
    query: str,
    candidates: list[ScoredChunk],
    *,
    max_tokens_context: int | None = None,
) -> RagResponse:
    """Turn retrieved chunks into context and generate a grounded answer."""
    settings = get_settings()
    completion_client = create_openai_client(settings)
    return answer_with_retrieval(
        query,
        candidates,
        completion_client,
        completion_model=settings.completion_model,
        max_tokens_context=max_tokens_context,
    )


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Process-wide Retriever so provider clients are built once, not per request."""
    settings = get_settings()
    return Retriever(
        embed_fn=create_embed_fn(settings),
        rerank_fn=create_rerank_fn(settings),
        rerank_expansion_factor=settings.rerank_expansion_factor,
    )


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


def create_rerank_fn(settings: Settings) -> RerankServiceFn:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    rerank_url = f"{settings.openrouter_base_url.rstrip('/')}/rerank"
    model = settings.rerank_model

    def rerank_fn(
        query: str, top_k: int, chunks: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        if not chunks:
            return []

        response = httpx.post(
            rerank_url,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "query": query,
                "documents": [item.chunk.content for item in chunks],
                "top_n": top_k,
            },
            timeout=30.0,
        )
        response.raise_for_status()

        reranked: list[ScoredChunk] = []
        for result in response.json()["results"]:
            # Re-map the index to the original chunk
            original = chunks[result["index"]]
            reranked.append(
                ScoredChunk(
                    chunk=original.chunk,
                    score=result["relevance_score"],
                    retrieval_score=original.retrieval_score,
                    rerank_score=result["relevance_score"],
                )
            )
        return reranked

    return rerank_fn


def create_markdown_processor(settings: Settings) -> MarkdownProcessor:
    """Create a RAG processor from application settings."""
    return MarkdownProcessor(
        create_openai_client(settings),
        settings.embedding_model,
        chunk_max_tokens=settings.chunk_max_tokens,
        chunk_min_tokens=settings.chunk_min_tokens,
        chunk_overlap_percent=settings.chunk_overlap_percent,
    )
