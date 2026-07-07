"""System user provisioning and tenant management."""

from sqlalchemy.orm import Session

from app.auth import generate_api_key, hash_api_key
from app.config import get_settings
from app.models import SystemUser


class SystemUserLookupError(Exception):
    pass


def create_system_user(
    db: Session, *, name: str, ratelimit: int = 100
) -> tuple[SystemUser, str]:
    """Create a system user and return the model plus the plaintext API key (shown once)."""
    settings = get_settings()
    api_key = generate_api_key()
    user = SystemUser(
        name=name,
        ratelimit=ratelimit,
        api_key_hash=hash_api_key(api_key, settings.api_key_hash_secret),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user, api_key


def get_system_user_by_name(db: Session, name: str) -> SystemUser:
    users = db.query(SystemUser).filter(SystemUser.name == name).all()
    if not users:
        raise SystemUserLookupError(f"No system user found with name '{name}'")
    if len(users) > 1:
        raise SystemUserLookupError(f"Multiple system users found with name '{name}'")
    return users[0]

