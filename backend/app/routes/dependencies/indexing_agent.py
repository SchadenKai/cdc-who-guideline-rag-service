from functools import lru_cache

from langgraph.graph.state import CompiledStateGraph

from app.agent.indexing.main import agent


@lru_cache
def get_indexing_agent() -> CompiledStateGraph:
    return agent
