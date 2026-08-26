"""Retrieve and query endpoints: POST /v1/retrieve and POST /v1/query.

Tenant identity comes from the authenticated SystemUser (API key).
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_system_user
from app.database import get_db
from app.models import SystemUser
from app.schemas.common import ErrorResponse
from app.schemas.query import QueryRequest, RagResponse, RetrieveRequest, RetrievalResult
from app.services.rag import answer_service, chunks_to_retrieval_result, retrieval_service

router = APIRouter(
    tags=["query"],
    dependencies=[Depends(get_current_system_user)],
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)


def _filters_dict(body: RetrieveRequest) -> dict | None:
    return body.filters.model_dump() if body.filters else None


@router.post(
    "/retrieve",
    response_model=RetrievalResult,
    summary="Retrieve ranked chunks for a query",
    operation_id="retrieve",
)
def retrieve(
    body: RetrieveRequest,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> RetrievalResult:
    start = time.monotonic()
    candidates = retrieval_service(
        query=body.query,
        top_k=body.top_k,
        filters=_filters_dict(body),
        user=user,
        collection_slug=body.collection_slug,
        use_rerank=body.use_rerank,
        session=db,
    )
    elapsed_ms = int((time.monotonic() - start) * 1000)
    return chunks_to_retrieval_result(
        body.query,
        candidates,
        top_k=body.top_k,
        use_rerank=body.use_rerank,
        latency_ms=elapsed_ms,
    )


@router.post(
    "/query",
    response_model=RagResponse,
    summary="Retrieve chunks and generate a grounded answer",
    operation_id="query",
)
def query(
    body: QueryRequest,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> RagResponse:
    candidates = retrieval_service(
        query=body.query,
        top_k=body.top_k,
        filters=_filters_dict(body),
        user=user,
        collection_slug=body.collection_slug,
        use_rerank=body.use_rerank,
        session=db,
    )
    return answer_service(
        body.query,
        candidates,
        max_tokens_context=body.max_tokens_context,
    )
