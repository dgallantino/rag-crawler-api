"""Bearer token authentication dependency for the trusted internal /v1/query endpoint."""

from __future__ import annotations

import hmac

from fastapi import Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.exceptions import AuthenticationError

_bearer_scheme = HTTPBearer(auto_error=False)


async def require_bearer_token(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    """FastAPI dependency that validates the internal service bearer token.

    Performs constant-time comparison to prevent timing attacks.
    Raises AuthenticationError (→ 401) on any failure.
    """
    settings = get_settings()

    if not authorization:
        raise AuthenticationError("Missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthenticationError("Invalid Authorization header format; expected 'Bearer <token>'")

    if not settings.internal_bearer_token:
        raise AuthenticationError("Internal bearer token is not configured")

    if not hmac.compare_digest(token, settings.internal_bearer_token):
        raise AuthenticationError("Invalid bearer token")
