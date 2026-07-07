"""Document upload and status API routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies.auth import get_current_system_user
from app.models import SystemUser
from app.schemas.documents import (
    DocumentStatusResponse,
    DocumentUploadResponse,
    DocumentValidationErrorResponse,
)
from app.services.document_validation import validate_markdown_upload
from app.services.documents import (
    DocumentConflictError,
    DocumentNotFoundError,
    create_document_upload,
    get_document_status,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={422: {"model": DocumentValidationErrorResponse}},
)
async def upload_document(
    file: UploadFile = File(...),
    user: SystemUser = Depends(get_current_system_user),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    content = await file.read()
    filename = file.filename or "unknown.md"

    validation = validate_markdown_upload(filename, content)
    if not validation.valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"accepted": False, "reason": validation.reason},
        )

    try:
        document = create_document_upload(
            db,
            user,
            filename,
            content.decode("utf-8"),
        )
    except DocumentConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return DocumentUploadResponse(
        document_id=document.id,
        filename=filename,
        accepted=True,
    )


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
def document_status(
    document_id: UUID,
    user: SystemUser = Depends(get_current_system_user),
    db: Session = Depends(get_db),
) -> DocumentStatusResponse:
    try:
        return get_document_status(db, user, document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        ) from exc
