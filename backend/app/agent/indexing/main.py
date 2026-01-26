from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

from .edges import is_chunked_docs_empty
from .nodes import (
    chunker_node,
    doc_builder_node,
    indexing_node,
    metadata_builder_node,
    web_scrapper,
)
from .state import AgentState

graph = StateGraph(state_schema=AgentState)

graph.add_node("web_scrapper", web_scrapper)
graph.add_node("metadata_builder_node", metadata_builder_node)
graph.add_node("chunker_node", chunker_node)
graph.add_node("doc_builder_node", doc_builder_node)
graph.add_node("indexing_node", indexing_node)

graph.add_node("is_chunked_docs_empty", is_chunked_docs_empty)

graph.set_entry_point("web_scrapper")
graph.set_finish_point("indexing_node")

graph.add_edge("web_scrapper", "chunker_node")
graph.add_edge("chunker_node", "metadata_builder_node")
graph.add_edge("metadata_builder_node", "is_chunked_docs_empty")
graph.add_edge("doc_builder_node", "indexing_node")

checkpointer = InMemorySaver()

agent = graph.compile(checkpointer=checkpointer)
