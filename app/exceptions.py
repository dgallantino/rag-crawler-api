"""Custom exception classes and FastAPI exception handlers for the RAG query API."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.common import ErrorResponse


class UnauthorizedError(Exception):
    """Raised when the request lacks a valid API key (→ 401)."""

    def __init__(self, message: str = "Invalid API key") -> None:
        super().__init__(message)
        self.message = message


class ForbiddenError(Exception):
    """Raised when the tenant lacks access to the requested resource (→ 403)."""

    def __init__(self, message: str = "Access denied") -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(Exception):
    """Raised when a requested resource does not exist (→ 404)."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message)
        self.message = message


class ConflictError(Exception):
    """Raised when a create/update would violate uniqueness (→ 409)."""

    def __init__(self, message: str = "Resource already exists") -> None:
        super().__init__(message)
        self.message = message


class ValidationFailedError(Exception):
    """Raised when business-level validation fails beyond schema (→ 422)."""

    def __init__(self, message: str = "Validation failed") -> None:
        super().__init__(message)
        self.message = message


class NotImplementedAPIError(Exception):
    """Raised by stub routes that are not yet wired (→ 501)."""

    def __init__(self, message: str = "Not implemented") -> None:
        super().__init__(message)
        self.message = message


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _format_validation_errors(exc: RequestValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", ()) if item != "body")
        msg = err.get("msg", "invalid")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "Validation failed"


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to the FastAPI app."""

    @app.exception_handler(UnauthorizedError)
    async def handle_unauthorized(request: Request, exc: UnauthorizedError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(
                error="unauthorized",
                message=exc.message,
                request_id=_request_id(request),
            ).model_dump(),
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(ForbiddenError)
    async def handle_forbidden(request: Request, exc: ForbiddenError) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content=ErrorResponse(
                error="forbidden",
                message=exc.message,
                request_id=_request_id(request),
            ).model_dump(),
        )

    @app.exception_handler(NotFoundError)
    async def handle_not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="not_found",
                message=exc.message,
                request_id=_request_id(request),
            ).model_dump(),
        )

    @app.exception_handler(ConflictError)
    async def handle_conflict(request: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=ErrorResponse(
                error="conflict",
                message=exc.message,
                request_id=_request_id(request),
            ).model_dump(),
        )

    @app.exception_handler(ValidationFailedError)
    async def handle_validation(request: Request, exc: ValidationFailedError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                message=exc.message,
                request_id=_request_id(request),
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=ErrorResponse(
                error="validation_error",
                message=_format_validation_errors(exc),
                request_id=_request_id(request),
            ).model_dump(),
        )

    @app.exception_handler(NotImplementedAPIError)
    async def handle_not_implemented(
        request: Request, exc: NotImplementedAPIError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            content=ErrorResponse(
                error="not_implemented",
                message=exc.message,
                request_id=_request_id(request),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unhandled(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error",
                message="An unexpected error occurred",
                request_id=_request_id(request),
            ).model_dump(),
        )
