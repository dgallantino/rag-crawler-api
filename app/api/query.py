"""Backend-facing retrieval endpoint: POST /v1/query.

Trusted internal callers authenticate with a shared bearer token and supply
all parameters explicitly, including user_id. This endpoint passes through to
retrieve() without applying any tenant-level scoping or rate limiting.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, Request

from app.dependencies.bearer import require_bearer_token
from app.schemas.query import BackendQueryRequest, RetrievalResult
from app.services.rag import retrieve

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["query"])


@router.post(
    "/query",
    response_model=RetrievalResult,
    dependencies=[Depends(require_bearer_token)],
    summary="Backend-facing retrieval query",
    operation_id="queryBackend",
)
def query_backend(body: BackendQueryRequest, request: Request) -> RetrievalResult:
    """Full-parameter retrieval for trusted internal/backend callers."""
    request_id = getattr(request.state, "request_id", None)
    start = time.monotonic()

    filters = body.filters.model_dump(exclude_none=True) if body.filters else None

    result = retrieve(
        user_id=body.user_id,
        query=body.query,
        top_k=body.top_k,
        filters=filters,
        rerank=body.rerank,
        collection=body.collection,
    )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "query_backend",
        extra={
            "request_id": request_id,
            "endpoint": "/v1/query",
            "user_id": body.user_id,
            "latency_ms": elapsed_ms,
            "status_code": 200,
        },
    )

    return result
