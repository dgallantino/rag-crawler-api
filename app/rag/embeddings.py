"""OpenRouter embedding client via OpenAI SDK."""

from openai import OpenAI

from app.config import get_settings

BATCH_SIZE = 100


def _get_client() -> OpenAI:
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    return OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    settings = get_settings()
    client = _get_client()
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        sorted_data = sorted(response.data, key=lambda item: item.index)
        embeddings.extend(item.embedding for item in sorted_data)

    return embeddings
