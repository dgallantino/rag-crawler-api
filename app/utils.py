"""Planned: Shared utility helpers used across the application."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def normalize_url(url: str) -> str:
    """Placeholder: normalize and validate URLs before crawling."""
    return url.strip().rstrip("/")
