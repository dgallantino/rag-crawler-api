"""Event types emitted by the RAG pipeline."""

from dataclasses import dataclass
from typing import Callable, Literal

StepName = Literal["chunking", "embedding", "storing", "failed"]
StepStatus = Literal["in_progress", "completed",]


@dataclass(frozen=True)
class DocumentStatusEvent:
    document_id: str
    step: StepName
    status: StepStatus


DocumentStatusHandler = Callable[[DocumentStatusEvent], None]
