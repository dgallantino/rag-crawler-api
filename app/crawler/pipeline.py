from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Iterator, List, TypeVar


T = TypeVar("T")
logger = logging.getLogger(__name__)


class DataRetrieverError(Exception):
    ...


# =========================================================
# Core models
# =========================================================

@dataclass
class Item(Generic[T]):
    """
    Standard payload wrapper passed between stages.
    """
    data: T
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageContext:
    """
    Shared mutable state for the whole pipeline.
    """
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


# =========================================================
# Base stage
# =========================================================

class BaseStage(ABC):
    name: str | None = None

    @property
    def stage_name(self) -> str:
        return self.name or self.__class__.__name__

    def __call__(
        self,
        items: Iterable[Item[Any]],
        context: StageContext,
    ) -> Iterator[Item[Any]]:
        yield from self.run(items, context)

    @abstractmethod
    def run(
        self,
        items: Iterable[Item[Any]],
        context: StageContext,
    ) -> Iterator[Item[Any]]:
        raise NotImplementedError


# =========================================================
# Pipeline
# =========================================================


class Pipeline:
    def __init__(self, stages: List[BaseStage]) -> None:
        if not stages:
            raise ValueError("Pipeline requires at least one stage.")
        self.stages = stages

    @staticmethod
    def _safe_close(obj: object) -> None:
        close = getattr(obj, "close", None)
        if callable(close):
            close()

    def run(
        self,
        initial_items: Iterable[Item[Any]] | None = None,
        context: StageContext | None = None,
        stop_on_error: bool = False,
    ) -> List[Item[Any]]:
        context = context or StageContext()
        stream: Iterable[Item[Any]] = initial_items or [Item(data=None)]
        opened_streams: List[Iterable[Item[Any]]] = []

        try:
            for stage in self.stages:
                stream = stage(stream, context)
                opened_streams.append(stream)
            return list(stream)

        except Exception as e:
            if stop_on_error:
                logger.exception("Pipeline failed")

                for s in reversed(opened_streams):
                    try:
                        self._safe_close(s)
                    except Exception:
                        logger.exception("Failed to close stream during cleanup")

                raise DataRetrieverError("pipeline failed") from e

            raise

    def iter(
        self,
        initial_items: Iterable[Item[Any]] | None = None,
        context: StageContext | None = None,
    ) -> Iterator[Item[Any]]:
        context = context or StageContext()
        stream: Iterable[Item[Any]] = initial_items or [Item(data=None)]
        opened_streams: List[Iterable[Item[Any]]] = []

        for stage in self.stages:
            stream = stage(stream, context)
            opened_streams.append(stream)

        # yield from stream
        try:
            yield from stream
        except Exception as e:
            raise DataRetrieverError("pipeline failed") from e
        finally:
            for s in reversed(opened_streams):
                try:
                    self._safe_close(s)
                except Exception:
                    logger.exception("failed to close stream")
