"""Backend-facing retrieval endpoint: POST /v1/query.

Trusted internal callers authenticate with a shared bearer token and supply
all parameters explicitly, including user_id. Retrieval and answer generation
are delegated to rag_service().
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.stubs import raise_not_implemented
from app.database import get_db
from app.dependencies.bearer import require_bearer_token
from app.rag.generation import RagResponse
from app.schemas.query import BackendQueryRequest

router = APIRouter(tags=["query"])


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
    raise_not_implemented()
