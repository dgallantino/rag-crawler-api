"""Retrieve and query endpoints: POST /v1/retrieve and POST /v1/query.

Tenant identity comes from the authenticated SystemUser (API key).
Handlers remain 501 until C2.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import get_current_system_user
from app.api.stubs import raise_not_implemented
from app.database import get_db
from app.models import SystemUser
from app.schemas.common import ErrorResponse
from app.schemas.query import QueryRequest, RagResponse, RetrieveRequest, RetrievalResult

router = APIRouter(
    tags=["query"],
    dependencies=[Depends(get_current_system_user)],
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
    },
)


@router.post(
    "/retrieve",
    response_model=RetrievalResult,
    summary="Retrieve ranked chunks for a query",
    operation_id="retrieve",
)
def retrieve(
    body: RetrieveRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> RetrievalResult:
    raise_not_implemented()


@router.post(
    "/query",
    response_model=RagResponse,
    summary="Retrieve chunks and generate a grounded answer",
    operation_id="query",
)
def query(
    body: QueryRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> RagResponse:
    raise_not_implemented()
