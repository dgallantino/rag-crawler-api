"""Document upload and status API routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_system_user
from app.database import get_db
from app.exceptions import ValidationFailedError
from app.models import Collection, SystemUser
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
from app.services.collections import get_collection, get_collection_by_slug
from app.services.documents import (
    create_document_upload,
    delete_document,
    get_document,
    get_document_status,
    list_documents,
    update_document,
    validate_markdown_upload,
)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(get_current_system_user)],
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
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


def _resolve_collection(
    db: Session,
    user: SystemUser,
    collection_id: UUID | None,
    collection_slug: str | None,
) -> Collection:
    if collection_id is not None:
        return get_collection(db, user, collection_id)
    return get_collection_by_slug(db, user, collection_slug)[0]


def _accept_upload(
    db: Session,
    user: SystemUser,
    filename: str,
    content: bytes,
    collection_id: UUID | None,
    collection_slug: str | None,
) -> DocumentUploadResponse:
    collection = _resolve_collection(db, user, collection_id, collection_slug)
    validation = validate_markdown_upload(filename, content)
    if not validation.valid:
        raise ValidationFailedError(validation.reason or "Invalid content")

    document = create_document_upload(
        db, collection, filename, content.decode("utf-8")
    )
    return DocumentUploadResponse(
        document_id=document.id,
        filename=filename,
        accepted=True,
    )


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
async def upload_document(
    file: UploadFile = File(...),
    collection_id: UUID | None = Form(None),
    collection_slug: str | None = Form(None),
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> DocumentUploadResponse:
    _validate_multipart_collection(collection_id, collection_slug)
    filename = file.filename or ""
    content = await file.read()
    return _accept_upload(db, user, filename, content, collection_id, collection_slug)


@router.post(
    "/json",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
def upload_document_json(
    body: DocumentUploadRequest,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> DocumentUploadResponse:
    return _accept_upload(
        db,
        user,
        body.filename,
        body.content.encode("utf-8"),
        body.collection_id,
        body.collection_slug,
    )


@router.get(
    "",
    response_model=list[DocumentListItem],
    responses={404: {"model": ErrorResponse}},
)
def list_documents_route(
    collection_id: UUID | None = None,
    collection_slug: str | None = None,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> list[DocumentListItem]:
    if collection_id is not None and collection_slug is not None:
        raise ValidationFailedError(
            "Provide at most one of collection_id or collection_slug"
        )
    return list_documents(
        db, user, collection_id=collection_id, collection_slug=collection_slug
    )


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
def document_status(
    document_id: UUID,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> DocumentStatusResponse:
    return get_document_status(db, user, document_id)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_document_route(
    document_id: UUID,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> DocumentResponse:
    return get_document(db, user, document_id)


@router.patch(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse}},
)
def update_document_route(
    document_id: UUID,
    body: DocumentUpdateRequest,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> DocumentResponse:
    return update_document(
        db, user, document_id, title=body.title, content=body.content
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
def delete_document_route(
    document_id: UUID,
    db: Session = Depends(get_db),
    user: SystemUser = Depends(get_current_system_user),
) -> None:
    delete_document(db, user, document_id)
