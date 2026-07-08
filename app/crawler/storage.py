from __future__ import annotations

from typing import Any, Iterable, Iterator

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
