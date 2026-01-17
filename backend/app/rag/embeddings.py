from typing import Optional

from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings

from app.core.config import settings


def get_bi_encoder() -> Embeddings:
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.bi_encoder_model,
        dimensions=settings.vector_dim,
    )


class Embedding:
    _default_embedding_config = {
        "api_key": settings.openai_api_key,
        "model": settings.bi_encoder_model,
        "dimensions": settings.vector_dim,
    }

    def __init__(self):
        self.client = None

    def get_client(
        self,
        provider: Optional[str] = "openai",
        config: list[dict] = _default_embedding_config,
    ) -> Embeddings:
        if self.client is None:
            if provider == "openai":
                self.client = OpenAIEmbeddings(**config)
            elif provider == "gemini":
                self.client = GoogleGenerativeAIEmbeddings(**config)
            elif provider == "azure":
                self.client = AzureOpenAIEmbeddings(**config)
            # only make this available during development
            elif provider == "fake":
                self.client = FakeEmbeddings(**config)
            elif provider == "nomic":
                self.client = NomicEmbeddings(**config)
            else:
                self.client = OpenAIEmbeddings(**config)
        return self.client

    def embed_query(self, text: str) -> list[float]:
        # compute the number of tokens based on the model

        # get the model name and provider from settings

        # run the logic
        return self.client.embed_query(text)

    def embed_documents(self, documents: list[str]) -> list[float]:
        # compute the number of tokens based on the model

        # get the model name and provider from settings

        # run the logic
        return self.client.embed_documents(documents)
