"""Redis-backed job status tracking for document processing."""

import json
from datetime import datetime, timezone
from typing import Literal

import redis

from app.config import get_settings
from app.rag.events import DocumentStatusEvent

StepName = Literal["chunking", "embedding", "storing"]
StepStatus = Literal["pending", "in_progress", "completed", "failed"]

JOB_STATUS_TTL_SECONDS = 86400
REDIS_KEY_PREFIX = "doc:job:"

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)
    return _redis_client


def _job_key(document_id: str) -> str:
    return f"{REDIS_KEY_PREFIX}{document_id}"


def _default_steps() -> dict[str, str]:
    return {
        "chunking": "pending",
        "embedding": "pending",
        "storing": "pending",
    }


def set_job_step(document_id: str, step: StepName, step_status: StepStatus) -> None:
    client = _get_redis()
    key = _job_key(document_id)
    existing = client.get(key)
    if existing:
        payload = json.loads(existing)
        steps = payload.get("steps", _default_steps())
    else:
        payload = {"steps": _default_steps()}
        steps = payload["steps"]

    steps[step] = step_status
    payload["step"] = step
    payload["steps"] = steps
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    client.setex(key, JOB_STATUS_TTL_SECONDS, json.dumps(payload))


def get_job_status(document_id: str) -> dict | None:
    raw = _get_redis().get(_job_key(document_id))
    if raw is None:
        return None
    return json.loads(raw)


def delete_job_status(document_id: str) -> None:
    _get_redis().delete(_job_key(document_id))


def handle_document_status_event(event: DocumentStatusEvent) -> None:
    """Handlers for document processing events emitted by the RAG pipeline."""
    if event.step == "failed":
        delete_job_status(event.document_id)
        return

    if event.step == "storing" and event.status == "completed":
        set_job_step(event.document_id, "storing", "completed")
        delete_job_status(event.document_id)
        return

    set_job_step(event.document_id, event.step, event.status)