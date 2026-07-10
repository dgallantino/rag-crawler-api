"""Redis-backed cache for API key → SystemUser resolution.

Caches the minimal tenant fields needed per request (id, name, ratelimit,
api_key_hash) to avoid a DB round-trip on every authenticated call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from uuid import UUID

import redis as redis_lib
from sqlalchemy.orm import Session

from app.auth import hash_api_key
from app.config import get_settings
from app.models import SystemUser

logger = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = "apikey:"


@dataclass
class CachedTenant:
    """Lightweight tenant representation populated from Redis cache."""

    id: UUID
    name: str
    ratelimit: int
    api_key_hash: str


def _cache_key(api_key_hash: str) -> str:
    return f"{_CACHE_KEY_PREFIX}{api_key_hash}"


def _to_cache_payload(user: SystemUser) -> str:
    return json.dumps(
        {
            "id": str(user.id),
            "name": user.name,
            "ratelimit": user.ratelimit,
            "api_key_hash": user.api_key_hash,
        }
    )


def _from_cache_payload(raw: str) -> CachedTenant:
    data = json.loads(raw)
    return CachedTenant(
        id=UUID(data["id"]),
        name=data["name"],
        ratelimit=data["ratelimit"],
        api_key_hash=data["api_key_hash"],
    )


def get_redis_client() -> redis_lib.Redis:
    settings = get_settings()
    return redis_lib.from_url(settings.redis_url, decode_responses=True)


def get_or_fetch(api_key: str, db: Session) -> SystemUser | None:
    """Resolve an API key to a SystemUser, using Redis as a look-aside cache.

    Returns the full ORM object (fetched from DB when needed so relationships
    like collections remain accessible). Returns None if the key is invalid.
    """
    settings = get_settings()
    api_key_hash = hash_api_key(api_key, settings.api_key_hash_secret)
    key = _cache_key(api_key_hash)

    try:
        r = get_redis_client()
        cached = r.get(key)
        if cached:
            payload = _from_cache_payload(cached)
            user = db.query(SystemUser).filter(SystemUser.id == payload.id).one_or_none()
            return user
    except Exception:
        logger.warning("Redis cache unavailable; falling back to DB lookup", exc_info=True)

    user = db.query(SystemUser).filter(SystemUser.api_key_hash == api_key_hash).one_or_none()
    if user is None:
        return None

    try:
        r = get_redis_client()
        r.set(key, _to_cache_payload(user), ex=settings.api_key_cache_ttl)
    except Exception:
        logger.warning("Failed to write tenant to Redis cache", exc_info=True)

    return user


def invalidate(api_key: str) -> None:
    """Remove a cached API key entry (call after key rotation or user deletion)."""
    settings = get_settings()
    api_key_hash = hash_api_key(api_key, settings.api_key_hash_secret)
    try:
        r = get_redis_client()
        r.delete(_cache_key(api_key_hash))
    except Exception:
        logger.warning("Failed to invalidate tenant cache entry", exc_info=True)
