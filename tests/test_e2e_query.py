"""Live e2e tests for RAG services — disabled by default."""

from __future__ import annotations

import os
import time
import traceback
from typing import Any

import pytest

from app.config import get_settings
from app.services.rag import answer_service, create_embed_fn, retrieval_service
from tests.e2e_report import (
    build_report_path,
    dump_db_state,
    write_e2e_report,
)
from tests.rag_helpers import _make_chunk


def e2e_is_configured() -> bool:
    if os.getenv("RUN_E2E") != "1":
        return False
    get_settings.cache_clear()
    return bool(get_settings().openrouter_api_key.strip())


@pytest.mark.e2e
@pytest.mark.skipif(
    not e2e_is_configured(),
    reason="Live e2e disabled. Set RUN_E2E=1 and OPENROUTER_API_KEY in .env.",
)
def test_rag_service_live_openrouter(
    db_session,
    test_user,
    test_collection,
) -> None:
    settings = get_settings()
    user, _ = test_user
    seed_content = (
        "Enterprise customers receive a 99.9% uptime SLA with 24/7 support."
    )
    query = "What uptime SLA do enterprise customers receive?"

    embed_fn = create_embed_fn(settings)
    chunk_vector = embed_fn(seed_content)
    _make_chunk(
        db_session,
        test_collection,
        content=seed_content,
        metadata={"doc_type": "contract"},
        chunk_vector=chunk_vector,
    )

    report: dict[str, Any] = {
        "report_type": "e2e_test",
        "test_name": "test_rag_service_live_openrouter",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "failed",
        "duration_ms": 0,
        "request": {
            "service": "rag",
            "query": query,
            "top_k": 3,
            "rerank": False,
            "collection_id": str(test_collection.id),
            "user_id": str(user.id),
        },
        "response": None,
        "database": {},
        "assertions": [
            "answer mentions 99.9%",
            "len(sources) > 0",
        ],
    }

    report_path = build_report_path(
        settings.e2e_report_dir,
        "test_rag_service_live_openrouter",
    )
    started = time.monotonic()

    try:
        report["database"]["before_request"] = dump_db_state(db_session)

        candidates = retrieval_service(
            query=query,
            top_k=3,
            filters=None,
            user=user,
            collection_slug="test-collection",
            use_rerank=False,
            session=db_session,
        )
        result = answer_service(query=query, candidates=candidates)

        report["response"] = {
            "answer": result.answer,
            "sources": [source.model_dump() for source in result.sources],
            "retrieved_chunks": len(candidates),
        }
        report["database"]["after_request"] = dump_db_state(db_session)

        assert "99.9%" in result.answer
        assert len(result.sources) > 0
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["duration_ms"] = int((time.monotonic() - started) * 1000)
        write_e2e_report(report_path, report)
