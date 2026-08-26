"""Collection management API routes."""

from uuid import UUID

from fastapi import APIRouter, Body, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_system_user
from app.database import get_db
from app.models import SystemUser
from app.schemas.collections import (
    CollectionCreateRequest,
    CollectionResponse,
    CollectionUpdateRequest,
)
from app.schemas.common import ErrorResponse
from app.schemas.examples import OPENAPI_EXAMPLES
from app.services.collections import (
    create_collection,
    delete_collection,
    get_collection,
    list_collections,
    update_collection,
)

router = APIRouter(
    prefix="/collections",
    tags=["collections"],
    dependencies=[Depends(get_current_system_user)],
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
    summary="Create a named collection for the authenticated tenant",
    operation_id="create_collection",
)
def create_collection_route(
    body: CollectionCreateRequest = Body(
        openapi_examples=OPENAPI_EXAMPLES["create_collection"],
    ),
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> CollectionResponse:
    return create_collection(db, user, name=body.name, slug=body.slug)


@router.get(
    "",
    response_model=list[CollectionResponse],
    summary="List collections owned by the authenticated tenant",
    operation_id="list_collections",
)
def list_collections_route(
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> list[CollectionResponse]:
    return list_collections(db, user)


@router.get(
    "/{collection_id}",
    response_model=CollectionResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get a collection by ID",
    operation_id="get_collection",
)
def get_collection_route(
    collection_id: UUID,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> CollectionResponse:
    return get_collection(db, user, collection_id)


@router.patch(
    "/{collection_id}",
    response_model=CollectionResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    summary="Update a collection name or slug",
    operation_id="update_collection",
)
def update_collection_route(
    collection_id: UUID,
    body: CollectionUpdateRequest = Body(
        openapi_examples=OPENAPI_EXAMPLES["update_collection"],
    ),
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> CollectionResponse:
    return update_collection(
        db, user, collection_id, name=body.name, slug=body.slug
    )


@router.delete(
    "/{collection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
    summary="Delete a collection and its documents",
    operation_id="delete_collection",
)
def delete_collection_route(
    collection_id: UUID,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> None:
    delete_collection(db, user, collection_id)
