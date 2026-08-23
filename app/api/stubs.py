"""Shared helpers for temporarily disabled API routes."""

from app.exceptions import NotImplementedAPIError


def raise_not_implemented(detail: str = "Not implemented") -> None:
    raise NotImplementedAPIError(detail)
