"""Tests for document CLI commands."""

import argparse
import json
from unittest.mock import MagicMock, patch

from app.cli import cmd_document_status, cmd_upload_document
from app.models import Document
from app.services.system_user import create_system_user


@patch("app.services.documents.trigger_process_document")
def test_cli_upload_document(mock_trigger, tmp_path, db_session, monkeypatch, capsys) -> None:
    monkeypatch.setattr("app.cli.SessionLocal", MagicMock(return_value=db_session))
    create_system_user(db_session, name="Dev User")

    md_file = tmp_path / "notes.md"
    md_file.write_text("# Notes\n\nHello.", encoding="utf-8")

    args = argparse.Namespace(path=str(md_file), name="Dev User")
    assert cmd_upload_document(args) == 0
    mock_trigger.assert_called_once()

    output = json.loads(capsys.readouterr().out)
    assert output["accepted"] is True
    assert output["filename"] == "notes.md"


def test_cli_upload_document_invalid_file(tmp_path, db_session, monkeypatch, capsys) -> None:
    monkeypatch.setattr("app.cli.SessionLocal", MagicMock(return_value=db_session))
    create_system_user(db_session, name="Dev User")

    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello", encoding="utf-8")

    args = argparse.Namespace(path=str(bad_file), name="Dev User")
    assert cmd_upload_document(args) == 1
    assert "Only .md files are accepted" in capsys.readouterr().err


def test_cli_document_status(db_session, monkeypatch, capsys) -> None:
    monkeypatch.setattr("app.cli.SessionLocal", MagicMock(return_value=db_session))
    user, _ = create_system_user(db_session, name="Dev User")
    document = Document(
        system_user_id=user.id,
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
