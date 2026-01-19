from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_nebius.chat_models import ChatNebius
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.logger import app_logger


def get_llm_provider() -> BaseChatModel:
    # llm provider must be retrieved from the database that is cached
    provider = settings.llm_provider

    if provider == "openai":
        return ChatOpenAI(model=settings.llm_model_name, api_key=settings.llm_api_key)
    elif provider == "anthropic":
        return ChatAnthropic(
            model_name=settings.llm_model_name, api_key=settings.llm_api_key
        )
    elif provider == "nebius":
        return ChatNebius(
            api_key=settings.llm_api_key,
            model=settings.llm_model_name,
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            api_key=settings.llm_api_key, model=settings.llm_model_name
        )
    return ChatOpenAI(model=settings.llm_model_name)


def test_chat_model():
    chat_llm = get_llm_provider()
    res = chat_llm.invoke("")
    app_logger.info(f"Testing chat model results: {res.content}")