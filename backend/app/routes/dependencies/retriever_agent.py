from functools import lru_cache

from backend.app.agent.retriever.main import agent
from langgraph.graph.state import CompiledStateGraph


@lru_cache
def get_retriever_agent() -> CompiledStateGraph:
    return agent
