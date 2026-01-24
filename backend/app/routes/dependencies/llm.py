from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings
from app.routes.dependencies.settings import get_app_settings
from app.services.llm.factory import ChatModelService


@lru_cache
def get_chat_model_service(
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> ChatModelService:
    return ChatModelService(
        provider=settings.llm_provider,
        model_name=settings.llm_model_name,
        api_key=settings.llm_api_key,
    )
