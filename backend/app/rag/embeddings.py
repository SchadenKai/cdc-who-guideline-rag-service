from functools import lru_cache

from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings

from app.core.config import settings

# TODO: Move this as a DI instead
from app.logger import logger
from app.services.llm.tokenizer import Tokenizer


def get_bi_encoder() -> Embeddings:
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.bi_encoder_model,
        dimensions=settings.vector_dim,
    )


class Embedding:
    def __init__(self, provider, config):
        self._client: Embeddings | None = None
        self.provider: str = provider
        self.config: dict = config

    def get_client(self) -> Embeddings:
        if self._client is None:
            if self.provider == "openai":
                self._client = OpenAIEmbeddings(**self.config)
            elif self.provider == "gemini":
                self._client = GoogleGenerativeAIEmbeddings(**self.config)
            elif self.provider == "azure":
                self._client = AzureOpenAIEmbeddings(**self.config)
            # only make this available during development
            elif self.provider == "fake":
                self._client = FakeEmbeddings(**self.config)
            elif self.provider == "nomic":
                self._client = NomicEmbeddings(**self.config)
            else:
                self._client = OpenAIEmbeddings(**self.config)
        return self._client

    def embed_query(self, text: str, tokenizer: Tokenizer) -> list[float]:
        # compute the number of tokens based on the model
        token_cnt = tokenizer.compute_token_cnt(text)
        logger.info(f"Total token count: {token_cnt}")
        # get the model name and provider from settings

        # run the logic
        embed = self.get_client()
        return embed.embed_query(text)

    def embed_documents(self, documents: list[str]) -> list[float]:
        # compute the number of tokens based on the model

        # get the model name and provider from settings

        # run the logic
        embed = self.get_client()
        return embed.embed_documents(documents)


@lru_cache
def get_embedding() -> Embedding:
    _config = {
        "api_key": settings.openai_api_key,
        "model": settings.bi_encoder_model,
        "dimensions": settings.vector_dim,
    }
    return Embedding(provider=settings.llm_provider, config=_config)
