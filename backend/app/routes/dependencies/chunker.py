from functools import lru_cache

from app.rag.chunker import ChunkerService


@lru_cache
def get_chunker() -> ChunkerService:
    return ChunkerService()
