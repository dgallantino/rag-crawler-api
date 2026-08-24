"""Collection management API routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_system_user
from app.api.stubs import raise_not_implemented
from app.database import get_db
from app.models import SystemUser
from app.schemas.collections import (
    CollectionCreateRequest,
    CollectionResponse,
    CollectionUpdateRequest,
)
from app.schemas.common import ErrorResponse

router = APIRouter(
    prefix="/collections",
    tags=["collections"],
    dependencies=[Depends(get_current_system_user)],
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
    },
)


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
)
def create_collection_route(
    body: CollectionCreateRequest,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> CollectionResponse:
    raise_not_implemented()


@router.get("", response_model=list[CollectionResponse])
def list_collections_route(
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> list[CollectionResponse]:
    raise_not_implemented()


@router.get(
    "/{collection_id}",
    response_model=CollectionResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_collection_route(
    collection_id: UUID,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> CollectionResponse:
    raise_not_implemented()


@router.patch(
    "/{collection_id}",
    response_model=CollectionResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
def update_collection_route(
    collection_id: UUID,
    body: CollectionUpdateRequest,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> CollectionResponse:
    raise_not_implemented()


@router.delete(
    "/{collection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
def delete_collection_route(
    collection_id: UUID,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> None:
    raise_not_implemented()
