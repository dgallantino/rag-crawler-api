"""Token counting shared by chunking (processor) and context packing (generation)."""

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    return len(_get_encoding().encode(text))
