from functools import lru_cache

from langgraph.graph.state import CompiledStateGraph

from app.agent.react_agent.main import agent


@lru_cache
def get_react_agent() -> CompiledStateGraph:
    return agent
