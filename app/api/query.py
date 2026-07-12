"""Backend-facing retrieval endpoint: POST /v1/query.

Trusted internal callers authenticate with a shared bearer token and supply
all parameters explicitly, including user_id. This endpoint passes through to
retrieve() without applying any tenant-level scoping or rate limiting.
"""

from __future__ import annotations

import logging
import time
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from openai import RateLimitError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.dependencies.bearer import require_bearer_token
from app.models import SystemUser
from app.rag.rerank import rerank
from app.rag.retrieval import retrieve
from app.schemas.query import BackendQueryRequest, RetrievalResult
from app.services.collections import CollectionNotFoundError, get_collection_by_slug
from app.services.rag import create_embed_fn, create_rerank_fn, to_retrieval_result
from app.services.rate_limit import check_rate_limit
from app.services.tenant_cache import get_redis_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


_QUERY_RATE_LIMIT = 512


@router.post(
    "/query",
    response_model=RetrievalResult,
    dependencies=[Depends(require_bearer_token)],
    summary="Backend-facing retrieval query",
    operation_id="queryBackend",
)
def query_backend(
    body: BackendQueryRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> RetrievalResult:
    """Handle full-parameter retrieval for trusted backend/internal systems."""
    request_id = getattr(request.state, "request_id", None)
    rate_limit_key = f"query:{body.user_id}"

    try:
        redis = get_redis_client()
        allowed, retry_after = check_rate_limit(redis, rate_limit_key, _QUERY_RATE_LIMIT)
    except Exception:
        logger.warning("Rate limit check failed; allowing request", exc_info=True)
        allowed, retry_after = True, 0

    if not allowed:
        logger.warning(
            "rate_limit_hit",
            extra={"request_id": request_id, "system_user": str(body.user_id)},
        )
        raise RateLimitError(retry_after=retry_after)

    try:
        user_id = UUID(body.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid user_id") from exc

    user = db.get(SystemUser, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    collection_id: str | None = None
    if body.collection is not None:
        try:
            collection = get_collection_by_slug(db, user, body.collection)
        except CollectionNotFoundError:
            raise HTTPException(status_code=404, detail="Collection not found")
        collection_id = str(collection.id)

    start = time.monotonic()
    settings = get_settings()
    embed_fn = create_embed_fn(settings)
    filters = body.filters.model_dump(exclude_none=True) if body.filters else None
    initial_k = body.top_k if not body.rerank else body.top_k * 4

    candidates = retrieve(
        body.query,
        initial_k,
        filters,
        collection_id,
        session=db,
        embed_fn=embed_fn,
    )

    if body.rerank:
        candidates = rerank(
            body.query,
            candidates,
            body.top_k,
            rerank_service_fn=create_rerank_fn(settings),
        )
    else:
        candidates = candidates[: body.top_k]

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

    return to_retrieval_result(
        candidates,
        query_used=body.query,
        top_k=body.top_k,
        reranked=body.rerank,
        latency_ms=elapsed_ms,
    )
