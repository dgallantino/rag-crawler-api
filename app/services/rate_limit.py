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


def token_bucket_rate_limit(
    r: 'redis_lib.Redis',
    api_key_hash: str,
    limit: int,
) -> tuple[bool, int]:
    """Check whether the API key is within its rate limit.

    Uses a token bucket algorithm.
    Returns (allowed, retry_after_seconds).
    """
    # Token bucket configuration
    refill_rate = limit / _WINDOW_SECONDS  # tokens per second
    max_tokens = limit

    key_tokens = f"{_KEY_PREFIX}tb:{api_key_hash}:tokens"
    key_timestamp = f"{_KEY_PREFIX}tb:{api_key_hash}:ts"

    # Use a Lua script for atomic token bucket logic
    lua_script = """
    local tokens_key = KEYS[1]
    local ts_key = KEYS[2]
    local max_tokens = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local window = tonumber(ARGV[4])

    local tokens = tonumber(redis.call("GET", tokens_key))
    local last_ts = tonumber(redis.call("GET", ts_key))
    if tokens == nil then tokens = max_tokens end
    if last_ts == nil then last_ts = now end

    local delta = math.max(0, now - last_ts)
    local refill = delta * refill_rate
    tokens = math.min(max_tokens, tokens + refill)
    if tokens >= 1 then
        tokens = tokens - 1
        redis.call("SET", tokens_key, tokens, "EX", window)
        redis.call("SET", ts_key, now, "EX", window)
        return {1, 0}  -- allowed, no retry
    else
        local seconds_until_next = math.ceil((1 - tokens) / refill_rate)
        redis.call("SET", tokens_key, tokens, "EX", window)
        redis.call("SET", ts_key, now, "EX", window)
        return {0, seconds_until_next}
    end
    """

    now = int(time.time())
    try:
        allowed, retry_after = r.eval(
            lua_script,
            2,
            key_tokens,
            key_timestamp,
            max_tokens,
            refill_rate,
            now,
            _WINDOW_SECONDS,
        )
        return bool(allowed), int(retry_after)
    except Exception:
        logger.warning("Redis token bucket rate-limit check failed; allowing request", exc_info=True)
        return True, 0