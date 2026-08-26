"""Tests for Celery task configuration."""

from app.services.jobs import celery_app, run_process_document


def test_run_process_document_uses_late_ack() -> None:
    assert run_process_document.acks_late is True


def test_celery_app_requeues_tasks_on_worker_loss() -> None:
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1
