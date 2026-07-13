"""Helpers for writing live e2e test reports as readable YAML."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from sqlalchemy.orm import Session

from app.models import Collection, Document, DocumentChunk, SystemUser


def summarize_vector(vector: list[float] | None, head: int = 8) -> str | None:
    if vector is None:
        return None
    if len(vector) <= head:
        return str(vector)
    preview = ", ".join(f"{value:.6f}" for value in vector[:head])
    remaining = len(vector) - head
    return f"[{preview}, ... (+{remaining} more)]"


def _serialize_value(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _serialize_row(row: Any, *, vector_field: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for column in row.__table__.columns:
        value = getattr(row, column.name)
        if vector_field and column.name == vector_field:
            data["chunk_vector_summary"] = summarize_vector(value)
            continue
        data[column.name] = _serialize_value(value)
    return data


def dump_db_state(session: Session) -> dict[str, list[dict[str, Any]]]:
    return {
        "system_users": [_serialize_row(row) for row in session.query(SystemUser).all()],
        "collections": [_serialize_row(row) for row in session.query(Collection).all()],
        "documents": [_serialize_row(row) for row in session.query(Document).all()],
        "document_chunks": [
            _serialize_row(row, vector_field="chunk_vector")
            for row in session.query(DocumentChunk).all()
        ],
    }


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        lower_key = key.lower()
        if lower_key in {"authorization", "x-api-key"}:
            redacted[key] = "***redacted***"
        else:
            redacted[key] = value
    return redacted


def build_report_path(report_dir: str, test_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"e2e_query_{timestamp}_{test_name}.yaml"
    return Path(report_dir) / filename


def write_e2e_report(path: Path, report: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            report,
            handle,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    return path
