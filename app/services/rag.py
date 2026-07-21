from __future__ import annotations

import httpx
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.rag.generation import RagResponse, answer_with_retrieval, normalize_chunks
from app.rag.rerank import RerankServiceFn, RerankedChunk, rerank
from app.rag.processor import MarkdownProcessor
from app.rag.retrieval import EmbedFn, RetrievedChunk, retrieve
from app.schemas.query import ChunkSource, RetrievalChunk, RetrievalResult

from openai import OpenAI

settings = get_settings()

def retrieval_service(
    query: str,
    top_k: int,
    filters: dict | None,
    collection: str | None,
    *,
    use_rerank: bool = False,
    session: Session,
) -> list[RetrievedChunk] | list[RerankedChunk]:
    """Retrieve and optionally rerank relevant chunks for a query."""
    settings = get_settings()
    embed_fn = create_embed_fn(settings)

    initial_retrieve_k = top_k if not use_rerank else top_k * 4
    candidates = retrieve(
        query, initial_retrieve_k, filters, collection,
        session=session, embed_fn=embed_fn)

    if use_rerank:
        return rerank(query, candidates, top_k, rerank_service_fn=create_rerank_fn(settings))
    return candidates[:top_k]


def _chunk_to_retrieval_chunk(item: RetrievedChunk | RerankedChunk) -> RetrievalChunk:
    chunk = item.chunk
    meta = chunk.chunk_metadata or {}
    source = None
    if chunk.document is not None:
        source = ChunkSource(
            document=chunk.document.title,
            page=meta.get("page"),
            url=chunk.document.url,
        )
    score = getattr(item, "rerank_score", item.similarity_score)
    return RetrievalChunk(
        chunk_id=str(chunk.id),
        text=chunk.content,
        score=score,
        source=source,
    )


def chunks_to_retrieval_result(
    query: str,
    chunks: list[RetrievedChunk] | list[RerankedChunk],
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
    candidates: list[RetrievedChunk] | list[RerankedChunk],
) -> RagResponse:
    """Turn retrieved chunks into context and generate a grounded answer."""
    settings = get_settings()
    completion_client = create_openai_client(settings)
    return answer_with_retrieval(
        query,
        normalize_chunks(candidates),
        completion_client,
        completion_model=settings.completion_model,
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
        query: str, top_k: int, chunks: list[RetrievedChunk]
    ) -> list[RerankedChunk]:
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

        reranked: list[RerankedChunk] = []
        for result in response.json()["results"]:
            # Re-map the index to the original chunk
            original = chunks[result["index"]]
            reranked.append(
                RerankedChunk(
                    chunk=original.chunk,
                    rerank_score=result["relevance_score"],
                    similarity_score=original.similarity_score,
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


