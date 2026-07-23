"""Tests for document CLI commands."""

import argparse
import json
from unittest.mock import MagicMock, patch

from app.cli import cmd_document_status, cmd_upload_document
from app.models import Document
from app.services.collections import create_collection
from app.services.system_user import create_system_user


@patch("app.services.documents.trigger_process_document")
def test_cli_upload_document(mock_trigger, tmp_path, db_session, monkeypatch, capsys) -> None:
    monkeypatch.setattr("app.cli.SessionLocal", MagicMock(return_value=db_session))
    user, _ = create_system_user(db_session, name="Dev User")
    create_collection(db_session, user, name="Dev Docs", slug="dev-docs")

    md_file = tmp_path / "notes.md"
    md_file.write_text("# Notes\n\nHello.", encoding="utf-8")

    args = argparse.Namespace(
        path=[str(md_file)], name="Dev User", collection_slug="dev-docs", json=False
    )
    assert cmd_upload_document(args) == 0
    mock_trigger.assert_called_once()

    out = capsys.readouterr().out
    assert "accepted: notes.md" in out
    assert "document_id=" in out


@patch("app.services.documents.trigger_process_document")
def test_cli_upload_document_json(
    mock_trigger, tmp_path, db_session, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("app.cli.SessionLocal", MagicMock(return_value=db_session))
    user, _ = create_system_user(db_session, name="Dev User")
    create_collection(db_session, user, name="Dev Docs", slug="dev-docs")

    md_file = tmp_path / "notes.md"
    md_file.write_text("# Notes\n\nHello.", encoding="utf-8")

    args = argparse.Namespace(
        path=[str(md_file)], name="Dev User", collection_slug="dev-docs", json=True
    )
    assert cmd_upload_document(args) == 0
    mock_trigger.assert_called_once()

    output = json.loads(capsys.readouterr().out)
    assert output["errors"] == []
    assert len(output["documents"]) == 1
    assert output["documents"][0]["accepted"] is True
    assert output["documents"][0]["filename"] == "notes.md"


@patch("app.services.documents.trigger_process_document")
def test_cli_upload_multiple_documents(
    mock_trigger, tmp_path, db_session, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("app.cli.SessionLocal", MagicMock(return_value=db_session))
    user, _ = create_system_user(db_session, name="Dev User")
    create_collection(db_session, user, name="Dev Docs", slug="dev-docs")

    file_a = tmp_path / "a.md"
    file_b = tmp_path / "b.md"
    file_a.write_text("# A\n", encoding="utf-8")
    file_b.write_text("# B\n", encoding="utf-8")

    args = argparse.Namespace(
        path=[str(file_a), str(file_b)],
        name="Dev User",
        collection_slug="dev-docs",
        json=False,
    )
    assert cmd_upload_document(args) == 0
    assert mock_trigger.call_count == 2

    out = capsys.readouterr().out
    assert "accepted: a.md" in out
    assert "accepted: b.md" in out


@patch("app.services.documents.trigger_process_document")
def test_cli_upload_document_mixed_results(
    mock_trigger, tmp_path, db_session, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("app.cli.SessionLocal", MagicMock(return_value=db_session))
    user, _ = create_system_user(db_session, name="Dev User")
    create_collection(db_session, user, name="Dev Docs", slug="dev-docs")

    good = tmp_path / "good.md"
    bad = tmp_path / "bad.txt"
    good.write_text("# Good\n", encoding="utf-8")
    bad.write_text("not markdown", encoding="utf-8")

    args = argparse.Namespace(
        path=[str(good), str(bad)],
        name="Dev User",
        collection_slug="dev-docs",
        json=True,
    )
    assert cmd_upload_document(args) == 1
    mock_trigger.assert_called_once()

    output = json.loads(capsys.readouterr().out)
    assert len(output["documents"]) == 1
    assert output["documents"][0]["filename"] == "good.md"
    assert len(output["errors"]) == 1
    assert output["errors"][0]["filename"] == "bad.txt"
    assert "Only .md files are accepted" in output["errors"][0]["error"]


def test_cli_upload_document_invalid_file(tmp_path, db_session, monkeypatch, capsys) -> None:
    monkeypatch.setattr("app.cli.SessionLocal", MagicMock(return_value=db_session))
    user, _ = create_system_user(db_session, name="Dev User")
    create_collection(db_session, user, name="Dev Docs", slug="dev-docs")

    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello", encoding="utf-8")

    args = argparse.Namespace(
        path=[str(bad_file)], name="Dev User", collection_slug="dev-docs", json=False
    )
    assert cmd_upload_document(args) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: notes.txt: Only .md files are accepted" in captured.err


def test_cli_document_status(db_session, monkeypatch, capsys) -> None:
    monkeypatch.setattr("app.cli.SessionLocal", MagicMock(return_value=db_session))
    user, _ = create_system_user(db_session, name="Dev User")
    collection = create_collection(db_session, user, name="Dev Docs", slug="dev-docs")
    document = Document(
        collection_id=collection.id,
        url="file://notes.md",
        title="notes.md",
        content="# Notes",
        status="success",
    )
    db_session.add(document)
    db_session.commit()

    args = argparse.Namespace(document_id=str(document.id), name="Dev User")
    assert cmd_document_status(args) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "success"
