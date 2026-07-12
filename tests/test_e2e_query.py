"""Live e2e tests for POST /v1/query — disabled by default."""

from __future__ import annotations

import os
import time
import traceback
from typing import Any

import pytest

from app.config import get_settings
from app.services.rag import create_embed_fn
from tests.e2e_report import (
    build_report_path,
    dump_db_state,
    redact_headers,
    write_e2e_report,
)
from tests.test_rag_retrieval import _make_chunk


def e2e_is_configured() -> bool:
    if os.getenv("RUN_E2E") != "1":
        return False
    get_settings.cache_clear()
    return bool(get_settings().openrouter_api_key.strip())


def _bearer_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.e2e
@pytest.mark.skipif(
    not e2e_is_configured(),
    reason="Live e2e disabled. Set RUN_E2E=1 and OPENROUTER_API_KEY in .env.",
)
def test_query_backend_live_openrouter(
    client,
    db_session,
    test_user,
    test_collection,
) -> None:
    settings = get_settings()
    user, _ = test_user
    seed_content = (
        "Enterprise customers receive a 99.9% uptime SLA with 24/7 support."
    )

    embed_fn = create_embed_fn(settings)
    chunk_vector = embed_fn(seed_content)
    _make_chunk(
        db_session,
        test_collection,
        content=seed_content,
        metadata={"doc_type": "contract"},
        chunk_vector=chunk_vector,
    )

    request_body = {
        "user_id": str(user.id),
        "query": "What uptime SLA do enterprise customers receive?",
        "top_k": 3,
        "rerank": False,
        "collection": test_collection.slug,
    }
    auth_headers = _bearer_headers(settings.internal_bearer_token)

    report: dict[str, Any] = {
        "report_type": "e2e_test",
        "test_name": "test_query_backend_live_openrouter",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "failed",
        "duration_ms": 0,
        "request": {
            "method": "POST",
            "path": "/v1/query",
            "headers": redact_headers(auth_headers),
            "body": request_body,
        },
        "response": None,
        "database": {},
        "assertions": [
            "status_code == 200",
            "len(results) > 0",
            "top result mentions 99.9%",
        ],
    }

    report_path = build_report_path(
        settings.e2e_report_dir,
        "test_query_backend_live_openrouter",
    )
    started = time.monotonic()

    try:
        report["database"]["before_request"] = dump_db_state(db_session)

        response = client.post("/v1/query", json=request_body, headers=auth_headers)
        report["response"] = {
            "status_code": response.status_code,
            "body": response.json() if response.content else None,
        }
        report["database"]["after_request"] = dump_db_state(db_session)

        assert response.status_code == 200
        payload = response.json()
        assert len(payload["results"]) > 0
        assert "99.9%" in payload["results"][0]["text"]
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["duration_ms"] = int((time.monotonic() - started) * 1000)
        write_e2e_report(report_path, report)
