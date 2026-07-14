"""Shared helpers for temporarily disabled API routes."""

from fastapi import HTTPException, status


def raise_not_implemented(detail: str = "Not implemented") -> None:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=detail)
