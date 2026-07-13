"""Tests for system user auth, provisioning, and CLI."""

import argparse
import json
from unittest.mock import MagicMock

from app.auth import generate_api_key, hash_api_key, verify_api_key
from app.cli import cmd_create_system_user, main
from app.models import SystemUser
from app.services.system_user import create_system_user


def test_hash_api_key_is_deterministic(api_key_secret: str) -> None:
    api_key = "sample-api-key"
    assert hash_api_key(api_key, api_key_secret) == hash_api_key(api_key, api_key_secret)


def test_verify_api_key(api_key_secret: str) -> None:
    api_key = generate_api_key()
    stored = hash_api_key(api_key, api_key_secret)
    assert verify_api_key(api_key, stored, api_key_secret) is True
    assert verify_api_key("wrong-key", stored, api_key_secret) is False


def test_create_system_user_persists_hash_not_plaintext(
    db_session, api_key_secret: str
) -> None:
    user, api_key = create_system_user(db_session, name="Acme Corp", ratelimit=200)

    assert user.name == "Acme Corp"
    assert user.ratelimit == 200
    assert user.id is not None
    assert user.api_key_hash != api_key
    assert verify_api_key(api_key, user.api_key_hash, api_key_secret)

    stored = db_session.get(SystemUser, user.id)
    assert stored is not None
    assert stored.api_key_hash == user.api_key_hash


def test_cli_create_system_user(db_session, api_key_secret: str, monkeypatch, capsys) -> None:
    session_factory = MagicMock(return_value=db_session)
    monkeypatch.setattr("app.cli.SessionLocal", session_factory)

    args = argparse.Namespace(name="CLI Tenant", ratelimit=150, collection=None)
    assert cmd_create_system_user(args) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["name"] == "CLI Tenant"
    assert output["ratelimit"] == 150
    assert output["api_key"]
    assert output["id"]
    assert output["created_at"]
    assert output["collections"] == []


def test_cli_main_create_system_user(db_session, api_key_secret: str, monkeypatch, capsys) -> None:
    session_factory = MagicMock(return_value=db_session)
    monkeypatch.setattr("app.cli.SessionLocal", session_factory)

    assert main(["create-system-user", "--name", "Main Tenant"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["name"] == "Main Tenant"
    assert output["ratelimit"] == 100
    assert output["collections"] == []
