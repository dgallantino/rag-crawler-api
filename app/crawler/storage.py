from __future__ import annotations

from typing import Any, Iterable, Iterator
from uuid import UUID

from sqlalchemy.orm import Session

from .pipeline import BaseStage, Item, StageContext


class FileStorage(BaseStage):
    """
    Placeholder sink-ish stage.
    Later this may write to DB and still yield the item onward.
    """

    def run(
        self,
        items: Iterable[Item[Any]],
        context: StageContext,
    ) -> Iterator[Item[Any]]:
        for item in items:
            yield Item(
                data=item.data,
                meta={
                    **item.meta,
                    "stored": True,
                },
            )


class DBStorage(BaseStage):
    """
    Sink: Item[EmbeddingInput] → Item[EmbeddingInput] (persists to Document).

    Intended behaviour:
    - Upsert a Document row per crawled page URL for the given system_user_id
    - Set document.status to "pending"
    - Caller triggers trigger_process_document() after the pipeline completes
    """

    def __init__(self, db: Session, system_user_id: UUID) -> None:
        super().__init__()
        self.db = db
        self.system_user_id = system_user_id

    def run(
        self,
        items: Iterable[Item[Any]],
        context: StageContext,
    ) -> Iterator[Item[Any]]:
        for item in items:
            # TODO: upsert Document(url, title, content) and commit
            yield Item(
                data=item.data,
                meta={
                    **item.meta,
                    "stored": False,
                },
            )
