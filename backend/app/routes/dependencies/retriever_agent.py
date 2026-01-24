from functools import lru_cache

from langgraph.graph.state import CompiledStateGraph

from app.agent.retriever.main import agent


@lru_cache
def get_retriever_agent() -> CompiledStateGraph:
    return agent
