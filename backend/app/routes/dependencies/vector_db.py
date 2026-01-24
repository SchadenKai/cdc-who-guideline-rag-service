from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.rag.db import VectorClient
from app.rag.embeddings import EmbeddingService
from app.routes.dependencies.embedding import get_embedding
from app.routes.dependencies.tokenizer import get_tokenizer_service
from app.services.llm.tokenizer import TokenizerService


@lru_cache()
def get_vector_client(
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding)],
    tokenizer_service: Annotated[TokenizerService, Depends(get_tokenizer_service)],
) -> VectorClient:
    return VectorClient(embedding_service, tokenizer_service)
