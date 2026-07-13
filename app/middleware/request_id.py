"""Starlette middleware that injects a UUID request ID into every request/response."""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attaches a unique request ID to each request.

    - Reads X-Request-ID from the incoming headers if present; otherwise generates a UUID4.
    - Stores the ID on request.state.request_id for use in handlers and exception formatters.
    - Echoes the ID back in the X-Request-ID response header.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
