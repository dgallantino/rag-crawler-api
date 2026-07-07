"""FastAPI dependencies for authentication."""

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.auth import hash_api_key
from app.config import get_settings
from app.database import get_db
from app.models import SystemUser


def get_current_system_user(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> SystemUser:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    settings = get_settings()
    api_key_hash = hash_api_key(x_api_key, settings.api_key_hash_secret)
    user = db.query(SystemUser).filter(SystemUser.api_key_hash == api_key_hash).one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return user
