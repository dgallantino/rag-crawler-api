"""Tests for CLI-only crawler runner."""

from unittest.mock import MagicMock, patch

import pytest

from app.crawler.extractors import DOMChunker
from app.crawler.formatters import ChunkPrinter
from app.crawler.pipeline import Item, StageContext
from app.crawler.runner import count_crawl_results, run_crawl_debug
from app.crawler.schemas import PageChunk
from app.crawler.sources import JSCrawler


def test_run_crawl_debug_raises_on_empty_seed_urls() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_crawl_debug([])


@patch("app.crawler.runner.Pipeline")
def test_run_crawl_debug_builds_three_stage_pipeline(mock_pipeline_cls: MagicMock) -> None:
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = ["chunk output"]
    mock_pipeline_cls.return_value = mock_pipeline

    results = run_crawl_debug(["https://example.com"], max_pages=5, headless=False, job_id="job-1")

    mock_pipeline_cls.assert_called_once()
    stages = mock_pipeline_cls.call_args[0][0]
    assert len(stages) == 3
    assert isinstance(stages[0], JSCrawler)
    assert stages[0].max_pages == 5
    assert stages[0].headless is False
    assert isinstance(stages[1], DOMChunker)
    assert isinstance(stages[2], ChunkPrinter)

    context = mock_pipeline.run.call_args.kwargs["context"]
    assert isinstance(context, StageContext)
    assert context.get("seed_urls") == ["https://example.com"]
    assert context.get("job_id") == "job-1"
    assert results == ["chunk output"]


@patch("app.crawler.runner.Pipeline")
def test_run_crawl_debug_generates_job_id_when_missing(mock_pipeline_cls: MagicMock) -> None:
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = []
    mock_pipeline_cls.return_value = mock_pipeline

    run_crawl_debug(["https://example.com"])

    context = mock_pipeline.run.call_args.kwargs["context"]
    assert context.get("job_id") is not None


def test_count_crawl_results() -> None:
    chunk = PageChunk(
        chunk_index=0,
        url="https://example.com",
        text="hello",
        headings=[],
        title="Example",
        selector="p",
        char_count=5,
        word_count=1,
        node_names=["p"],
    )
    results = ["printed", Item(data=chunk, meta={}), Item(data={"page": 1}, meta={})]
    pages, chunks = count_crawl_results(results)
    assert pages == 1
    assert chunks == 2
