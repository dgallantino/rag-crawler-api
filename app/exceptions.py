"""Custom exception classes and FastAPI exception handlers for the RAG query API."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.schemas.query import ErrorResponse


class AuthenticationError(Exception):
    """Raised when credentials are missing or invalid (→ 401)."""

    def __init__(self, message: str = "Missing or invalid credentials") -> None:
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


class ValidationFailedError(Exception):
    """Raised when business-level validation fails beyond schema (→ 422)."""

    def __init__(self, message: str = "Validation failed") -> None:
        super().__init__(message)
        self.message = message


class RateLimitError(Exception):
    """Raised when the per-API-key rate limit is exceeded (→ 429)."""

    def __init__(self, retry_after: int = 60, message: str = "Too many requests") -> None:
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to the FastAPI app."""

    @app.exception_handler(AuthenticationError)
    async def handle_auth(request: Request, exc: AuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(
                error="unauthorized",
                message=exc.message,
                request_id=_request_id(request),
            ).model_dump(),
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

    @app.exception_handler(RateLimitError)
    async def handle_rate_limit(request: Request, exc: RateLimitError) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(exc.retry_after)},
            content=ErrorResponse(
                error="rate_limited",
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
