"""Collection management API routes."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.stubs import raise_not_implemented
from app.database import get_db
from app.dependencies.auth import get_current_system_user
from app.models import SystemUser
from app.schemas.collections import CollectionCreateRequest, CollectionCreateResponse

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
    raise_not_implemented()
