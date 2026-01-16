from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings

from app.core.config import settings


def get_bi_encoder() -> Embeddings:
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.bi_encoder_model,
        dimensions=settings.vector_dim,
    )
