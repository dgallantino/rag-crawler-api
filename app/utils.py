"""Planned: Shared utility helpers used across the application."""

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_url(url: str) -> str:
    """Placeholder: normalize and validate URLs before crawling."""
    return url.strip().rstrip("/")
