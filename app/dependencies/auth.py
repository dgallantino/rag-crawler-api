"""FastAPI dependencies for API key authentication."""

from __future__ import annotations

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from app.database import get_db
from app.exceptions import AuthenticationError
from app.models import SystemUser
from app.services import tenant_cache


def get_current_system_user(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> SystemUser:
    """Resolve an API key to a SystemUser, with Redis look-aside caching.

    Raises AuthenticationError (→ 401) when the header is missing or the key
    is not recognised.
    """
    if not x_api_key:
        raise AuthenticationError("Missing X-API-Key header")

    user = tenant_cache.get_or_fetch(x_api_key, db)
    if user is None:
        raise AuthenticationError("Invalid API key")

    return user
