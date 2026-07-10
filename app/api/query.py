"""Backend-facing retrieval endpoint: POST /v1/query.

Trusted internal callers authenticate with a shared bearer token and supply
all parameters explicitly, including user_id. This endpoint passes through to
retrieve() without applying any tenant-level scoping or rate limiting.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, Request
from openai import RateLimitError

from app.dependencies.bearer import require_bearer_token
from app.schemas.query import BackendQueryRequest, RetrievalResult
from app.services.rag import rag
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
def query_backend(body: BackendQueryRequest, request: Request) -> RetrievalResult:
    """
    Handle full-parameter retrieval for trusted backend/internal systems.

    This endpoint is intended to be called by other backend services or systems that are protected by
    a more advanced authentication layer—such as internal API gateways or service meshes—that
    implement stronger authorization, user validation, or per-tenant rate limiting. Unlike end-user
    endpoints, all retrieval parameters including user_id and filtering are provided directly by the caller,
    with the assumption that the caller operates in a secured network boundary and enforces its own
    authentication and authorization logic upstream.

    Common scenarios:
      - Invoked by internal applications behind a secure gateway.
      - Used in service-to-service communication within a trusted environment.
      - Fronted by systems that integrate with centralized authentication or SSO providers.

    Note: This endpoint does not do further tenant-level checks or rate limiting.
    """
    request_id = getattr(request.state, "request_id", None)
    rate_limit_key = f"query:{body.user_id}"

    # Rate limit check
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

    start = time.monotonic()

    filters = body.filters.model_dump(exclude_none=True) if body.filters else None

    result = rag(
        user_id=body.user_id,
        query=body.query,
        top_k=body.top_k,
        filters=filters,
        with_rerank=body.rerank,
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
