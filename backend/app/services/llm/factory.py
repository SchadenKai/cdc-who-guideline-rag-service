from langchain_openai import OpenAI
from langchain_core.language_models.llms import BaseLLM


def get_llm_provider() -> BaseLLM:
    return OpenAI()
