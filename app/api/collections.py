"""Collection management API routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies.auth import get_current_system_user
from app.models import SystemUser
from app.schemas.collections import CollectionCreateRequest, CollectionCreateResponse
from app.services.collections import CollectionConflictError, create_collection

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post(
    "",
    response_model=CollectionCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_collection_route(
    body: CollectionCreateRequest,
    user: SystemUser = Depends(get_current_system_user),
    db: Session = Depends(get_db),
) -> CollectionCreateResponse:
    try:
        collection = create_collection(db, user, name=body.name, slug=body.slug)
    except CollectionConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return CollectionCreateResponse(
        id=collection.id,
        name=collection.name,
        slug=collection.slug,
        created_at=collection.created_at,
    )
