"""Pydantic schemas for /v1/query and /v1/agent/search endpoints.

Mirrors the OpenAPI 3.1 contract in tmp-RAG-query-expected-shape.yaml exactly.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DateRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    after: date | None = None
    before: date | None = None


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any] | None = None
    owner_ref: str | None = None
    date_range: DateRange | None = None


class BackendQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, description="Tenant identifier supplied by the trusted caller")
    query: str = Field(min_length=1, description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")
    rerank: bool = Field(default=True, description="Whether to apply a reranking pass")
    collection: str | None = Field(default=None, description="Named collection to search within")
    filters: Filters | None = None
    max_tokens_context: int | None = Field(default=None, ge=1, description="Optional cap on total tokens returned")


class AgentSearchRequest(BaseModel):
    """Intentionally minimal. user_id, top_k, rerank, and filters are resolved server-side."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Natural language query")
    collection: str | None = Field(default=None, description="Optional collection slug to restrict search to")


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
    """Full retrieval result returned by /v1/query (debug fields included)."""

    results: list[RetrievalChunk] = Field(default_factory=list)
    query_used: str | None = None
    latency_ms: int | None = None
    top_k: int | None = None
    reranked: bool | None = None


class AgentRetrievalResult(BaseModel):
    """Trimmed result for /v1/agent/search — debug fields stripped."""

    results: list[RetrievalChunk] = Field(default_factory=list)
    query_used: str | None = None


class ErrorResponse(BaseModel):
    """Shared error envelope across all endpoints."""

    error: str
    message: str
    request_id: str | None = None
