"""Backend-facing retrieval endpoint: POST /v1/query.

Trusted internal callers authenticate with a shared bearer token and supply
all parameters explicitly, including user_id. Retrieval and answer generation
are delegated to rag_service().
"""

from __future__ import annotations

import logging
import time
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from openai import RateLimitError
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies.bearer import require_bearer_token
from app.models import SystemUser
from app.rag.generation import RagResponse
from app.schemas.query import BackendQueryRequest
from app.services.collections import CollectionNotFoundError, get_collection_by_slug
from app.services.rag import rag_service
from app.services.rate_limit import check_rate_limit
from app.services.tenant_cache import get_redis_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


_QUERY_RATE_LIMIT = 512


@router.post(
    "/query",
    response_model=RagResponse,
    dependencies=[Depends(require_bearer_token)],
    summary="Backend-facing retrieval query",
    operation_id="queryBackend",
)
def query_backend(
    body: BackendQueryRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> RagResponse:
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
    filters = body.filters.model_dump(exclude_none=True) if body.filters else None

    result = rag_service(
        query=body.query,
        top_k=body.top_k,
        filters=filters,
        collection=collection_id,
        use_rerank=body.rerank,
        session=db,
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
