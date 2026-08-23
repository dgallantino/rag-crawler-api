"""Document upload and status API routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.orm import Session

from app.api.stubs import raise_not_implemented
from app.database import get_db
from app.exceptions import ValidationFailedError
from app.schemas.common import ErrorResponse
from app.schemas.documents import (
    DocumentListItem,
    DocumentResponse,
    DocumentStatusResponse,
    DocumentUpdateRequest,
    DocumentUploadRequest,
    DocumentUploadResponse,
    validate_collection_identifier,
)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={
        422: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
    },
)


def _validate_multipart_collection(
    collection_id: UUID | None,
    collection_slug: str | None,
) -> None:
    try:
        validate_collection_identifier(collection_id, collection_slug)
    except ValueError as exc:
        raise ValidationFailedError(str(exc)) from exc


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={409: {"model": ErrorResponse}},
)
async def upload_document(
    file: UploadFile = File(...),
    collection_id: UUID | None = Form(None),
    collection_slug: str | None = Form(None),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    _validate_multipart_collection(collection_id, collection_slug)
    raise_not_implemented()


@router.post(
    "/json",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={409: {"model": ErrorResponse}},
)
def upload_document_json(
    body: DocumentUploadRequest,
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    raise_not_implemented()


@router.get("", response_model=list[DocumentListItem])
def list_documents_route(
    collection_id: UUID | None = None,
    collection_slug: str | None = None,
    db: Session = Depends(get_db),
) -> list[DocumentListItem]:
    if collection_id is not None and collection_slug is not None:
        raise ValidationFailedError(
            "Provide at most one of collection_id or collection_slug"
        )
    raise_not_implemented()


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
def document_status(
    document_id: UUID,
    db: Session = Depends(get_db),
) -> DocumentStatusResponse:
    raise_not_implemented()


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_document_route(
    document_id: UUID,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    raise_not_implemented()


@router.patch(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse}},
)
def update_document_route(
    document_id: UUID,
    body: DocumentUpdateRequest,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    raise_not_implemented()


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
def delete_document_route(
    document_id: UUID,
    db: Session = Depends(get_db),
) -> None:
    raise_not_implemented()
