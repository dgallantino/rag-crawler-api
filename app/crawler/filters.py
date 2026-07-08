from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator

from .pipeline import BaseStage, Item, StageContext
from .schemas import CrawlerData


logger = logging.getLogger(__name__)


class RemoveImageNames(BaseStage):
    """
    Some time pages only have one big image inside it
    those image renders to image name this stage remove it from pipeline
    """
    
    name = "RemoveImageNames"
    def __init__(self, *extensions):
        super().__init__()
        self._extensions = tuple(extensions)

    def is_an_image_name(self, name:str):
        endswith_img_ext = name.endswith(self._extensions)
        return endswith_img_ext
    def run(
        self,
        items: Iterable[Item[CrawlerData]],
        context: StageContext,
    ) -> Iterator[Item[Any]]:
        for item in items:

            if self.is_an_image_name(item.data.text):
                logger.warning("discarding %s" % item.data.url)
                continue

            yield Item(
                data=item.data,
                meta={
                    **item.meta,
                },
            )


class RemoveEmptyPages(BaseStage):
    name = "RemoveEmptyPages"
    def __init__(self, char_thresh=100):
        super().__init__()
        self.char_thresh=char_thresh

    def run(
        self,
        items: Iterable[Item[CrawlerData]],
        context: StageContext,
    ) -> Iterator[Item[Any]]:
        for item in items:
            if len(item.data.text) < self.char_thresh:
                logger.warning("discarding %s" % item.data.url)
                continue
            yield Item(data=item.data, meta=item.meta,)
