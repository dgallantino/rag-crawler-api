"""Document upload and status API routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.stubs import raise_not_implemented
from app.database import get_db
from app.schemas.documents import (
    DocumentStatusResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentValidationErrorResponse,
    validate_collection_identifier,
)

router = APIRouter(prefix="/documents", tags=["documents"])


def _validate_multipart_collection(
    collection_id: UUID | None,
    collection_slug: str | None,
) -> None:
    try:
        validate_collection_identifier(collection_id, collection_slug)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={422: {"model": DocumentValidationErrorResponse}},
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
)
def upload_document_json(
    body: DocumentUploadRequest,
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    raise_not_implemented()


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
def document_status(
    document_id: UUID,
    db: Session = Depends(get_db),
) -> DocumentStatusResponse:
    raise_not_implemented()
