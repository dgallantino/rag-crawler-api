"""FastAPI dependencies for API-key authentication."""

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.database import get_db
from app.exceptions import UnauthorizedError
from app.models import SystemUser
from app.services.system_user import get_system_user_by_api_key

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_system_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> SystemUser:
    if credentials is None or not credentials.credentials:
        raise UnauthorizedError("Invalid API key")

    user = get_system_user_by_api_key(db, credentials.credentials)
    if user is None:
        raise UnauthorizedError("Invalid API key")
    return user
