"""Redis sliding-window rate limiter for per-API-key request throttling.

Uses a fixed-window counter (60-second buckets) keyed by api_key_hash.
The limit is taken directly from SystemUser.ratelimit (requests per minute).
"""

from __future__ import annotations

import logging
import time

import redis as redis_lib

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60
_KEY_PREFIX = "ratelimit:"


def _rate_limit_key(api_key_hash: str) -> str:
    bucket = int(time.time()) // _WINDOW_SECONDS
    return f"{_KEY_PREFIX}{api_key_hash}:{bucket}"


def check_rate_limit(
    r: redis_lib.Redis,
    api_key_hash: str,
    limit: int,
) -> tuple[bool, int]:
    """Check whether the API key is within its rate limit.

    Uses a fixed-window (60 s) counter. The counter key expires automatically
    after the window so no manual cleanup is needed.

    Args:
        r: Redis client instance.
        api_key_hash: Hashed API key used as part of the Redis key.
        limit: Maximum requests allowed per minute for this tenant.

    Returns:
        (allowed, retry_after_seconds) — retry_after is 0 when allowed.
    """
    if limit <= 0:
        return True, 0

    key = _rate_limit_key(api_key_hash)

    try:
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, _WINDOW_SECONDS)
        count, _ = pipe.execute()
    except Exception:
        logger.warning("Redis rate-limit check failed; allowing request", exc_info=True)
        return True, 0

    if count > limit:
        seconds_elapsed = int(time.time()) % _WINDOW_SECONDS
        retry_after = _WINDOW_SECONDS - seconds_elapsed
        return False, retry_after

    return True, 0
