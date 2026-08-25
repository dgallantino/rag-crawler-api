"""Tests for the run-debug-server CLI command."""

import argparse
from unittest.mock import patch

from app.cli import cmd_run_debug_server, main


@patch("uvicorn.run")
def test_cmd_run_debug_server(mock_run) -> None:
    args = argparse.Namespace(host="127.0.0.1", port=8000, reload=False)
    assert cmd_run_debug_server(args) == 0
    mock_run.assert_called_once_with(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


@patch("uvicorn.run")
def test_cli_main_run_debug_server(mock_run) -> None:
    assert main(["run-debug-server", "--host", "0.0.0.0", "--port", "9000", "--reload"]) == 0
    mock_run.assert_called_once_with(
        "app.main:app",
        host="0.0.0.0",
        port=9000,
        reload=True,
    )
