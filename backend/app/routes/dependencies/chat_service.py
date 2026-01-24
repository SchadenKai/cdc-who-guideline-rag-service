from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph

from app.routes.dependencies.llm import get_chat_model_service
from app.routes.dependencies.react_agent import get_react_agent
from app.services.chat import ChatService
from app.services.llm.factory import ChatModelService


@lru_cache
def get_chat_service(
    agent: Annotated[CompiledStateGraph, Depends(get_react_agent)],
    chat_model_service: Annotated[ChatModelService, Depends(get_chat_model_service)],
) -> ChatService:
    return ChatService(agent=agent, chat_model_service=chat_model_service)
