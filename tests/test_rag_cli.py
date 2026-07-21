"""Tests for RAG CLI commands."""

from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock, patch

from app.cli import cmd_query, cmd_retrieve
from app.rag.generation import RagResponse
from app.rag.retrieval import RetrievedChunk
from app.services.collections import create_collection
from app.services.system_user import create_system_user


@patch("app.cli.answer_service")
@patch("app.cli.retrieval_service")
def test_cli_retrieve(mock_retrieval, mock_answer, db_session, monkeypatch, capsys) -> None:
    monkeypatch.setattr("app.cli.SessionLocal", MagicMock(return_value=db_session))
    user, _ = create_system_user(db_session, name="Dev User")
    create_collection(db_session, user, name="Dev Docs", slug="dev-docs")

    chunk = MagicMock()
    chunk.id = "chunk-1"
    chunk.content = "Enterprise SLA is 99.9% uptime"
    chunk.chunk_metadata = {"section_header": "SLA"}
    chunk.document = MagicMock(title="sla.md", url="file://sla.md")
    mock_retrieval.return_value = [RetrievedChunk(chunk=chunk, similarity_score=0.91)]

    args = argparse.Namespace(
        query="What is the SLA?",
        name="Dev User",
        collection_slug="dev-docs",
        top_k=3,
        rerank=False,
        filters=None,
        json=True,
    )
    assert cmd_retrieve(args) == 0
    mock_answer.assert_not_called()

    output = json.loads(capsys.readouterr().out)
    assert output["query_used"] == "What is the SLA?"
    assert output["top_k"] == 3
    assert output["reranked"] is False
    assert output["results"][0]["text"] == "Enterprise SLA is 99.9% uptime"
    assert output["results"][0]["score"] == 0.91

    mock_retrieval.assert_called_once_with(
        query="What is the SLA?",
        top_k=3,
        filters=None,
        user=user,
        collection_slug="dev-docs",
        use_rerank=False,
        session=db_session,
    )


@patch("app.cli.answer_service")
@patch("app.cli.retrieval_service")
def test_cli_query(mock_retrieval, mock_answer, db_session, monkeypatch, capsys) -> None:
    monkeypatch.setattr("app.cli.SessionLocal", MagicMock(return_value=db_session))
    user, _ = create_system_user(db_session, name="Dev User")
    create_collection(db_session, user, name="Dev Docs", slug="dev-docs")

    chunk = MagicMock()
    candidates = [RetrievedChunk(chunk=chunk, similarity_score=0.9)]
    mock_retrieval.return_value = candidates
    mock_answer.return_value = RagResponse(
        answer="Enterprise SLA is 99.9% uptime",
        sources=[],
    )

    args = argparse.Namespace(
        query="What is the SLA?",
        name="Dev User",
        collection_slug=None,
        top_k=5,
        rerank=True,
        filters='{"metadata": {"doc_type": "contract"}}',
    )
    assert cmd_query(args) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["answer"] == "Enterprise SLA is 99.9% uptime"
    mock_answer.assert_called_once_with("What is the SLA?", candidates)
    mock_retrieval.assert_called_once_with(
        query="What is the SLA?",
        top_k=5,
        filters={"metadata": {"doc_type": "contract"}},
        user=user,
        collection_slug=None,
        use_rerank=True,
        session=db_session,
    )


def test_cli_retrieve_invalid_filters(db_session, monkeypatch, capsys) -> None:
    monkeypatch.setattr("app.cli.SessionLocal", MagicMock(return_value=db_session))
    create_system_user(db_session, name="Dev User")

    args = argparse.Namespace(
        query="What is the SLA?",
        name="Dev User",
        collection_slug=None,
        top_k=5,
        rerank=False,
        filters="not-json",
    )
    assert cmd_retrieve(args) == 1
    assert "invalid JSON for filters" in capsys.readouterr().err
