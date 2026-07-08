from unittest.mock import patch

from app.rag.events import DocumentStatusEvent
from app.services.job_status import handle_document_status_event




"""Tests for document status event handlers."""

@patch("app.services.job_status.delete_job_status")
@patch("app.services.job_status.set_job_step")
def test_handle_document_status_event_in_progress(mock_set_step, mock_delete_status) -> None:
    event = DocumentStatusEvent("doc-1", "chunking", "in_progress")
    handle_document_status_event(event)

    mock_set_step.assert_called_once_with("doc-1", "chunking", "in_progress")
    mock_delete_status.assert_not_called()


@patch("app.services.job_status.delete_job_status")
@patch("app.services.job_status.set_job_step")
def test_handle_document_status_event_success_terminal(mock_set_step, mock_delete_status) -> None:
    event = DocumentStatusEvent("doc-1", "storing", "completed")
    handle_document_status_event(event)

    mock_set_step.assert_called_once_with("doc-1", "storing", "completed")
    mock_delete_status.assert_called_once_with("doc-1")


@patch("app.services.job_status.delete_job_status")
@patch("app.services.job_status.set_job_step")
def test_handle_document_status_event_failure_terminal(mock_set_step, mock_delete_status) -> None:
    event = DocumentStatusEvent("doc-1", "failed", "completed")
    handle_document_status_event(event)

    mock_set_step.assert_not_called()
    mock_delete_status.assert_called_once_with("doc-1")