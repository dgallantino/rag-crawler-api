"""Pydantic schemas for /v1/retrieve and /v1/query.

The HTTP contract is owned by these models. FastAPI generates OpenAPI at /docs.
Tenant identity is the authenticated SystemUser (API key); D1 wires auth.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.rag.generation import RagResponse, Source

__all__ = [
    "DateRange",
    "Filters",
    "RetrieveRequest",
    "QueryRequest",
    "ChunkSource",
    "RetrievalChunk",
    "RetrievalResult",
    "RagResponse",
    "Source",
]


class DateRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    after: date | None = None
    before: date | None = None


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any] | None = None
    date_range: DateRange | None = None


class RetrieveRequest(BaseModel):
    """Body for POST /v1/retrieve. Matches retrieval_service keyword names."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")
    use_rerank: bool = Field(
        default=False, description="Whether to apply a reranking pass"
    )
    collection_slug: str | None = Field(
        default=None, description="Named collection to search within"
    )
    filters: Filters | None = None


class QueryRequest(RetrieveRequest):
    """Body for POST /v1/query: retrieve params plus optional LLM context cap."""

    max_tokens_context: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap on total tokens included in the LLM context",
    )


class ChunkSource(BaseModel):
    document: str | None = None
    page: int | None = None
    url: str | None = None


class RetrievalChunk(BaseModel):
    chunk_id: str | None = None
    text: str | None = None
    score: float | None = None
    source: ChunkSource | None = None


class RetrievalResult(BaseModel):
    """Shaped retrieval result from chunks_to_retrieval_result."""

    results: list[RetrievalChunk] = Field(default_factory=list)
    query_used: str | None = None
    latency_ms: int | None = None
    top_k: int | None = None
    reranked: bool | None = None
