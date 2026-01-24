from functools import lru_cache

from backend.app.agent.indexing.main import agent
from langgraph.graph.state import CompiledStateGraph


@lru_cache
def get_indexing_agent() -> CompiledStateGraph:
    return agent
