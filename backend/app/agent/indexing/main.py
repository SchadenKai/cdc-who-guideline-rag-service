from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

from .nodes import chunker_node, doc_builder_node, document_loader, indexing_node
from .state import AgentState

graph = StateGraph(state_schema=AgentState)

graph.add_node("document_loader", document_loader)
graph.add_node("chunker_node", chunker_node)
graph.add_node("doc_builder_node", doc_builder_node)
graph.add_node("indexing_node", indexing_node)

graph.set_entry_point("document_loader")
graph.set_finish_point("indexing_node")

graph.add_edge("document_loader", "chunker_node")
graph.add_edge("chunker_node", "doc_builder_node")
graph.add_edge("doc_builder_node", "indexing_node")

checkpointer = InMemorySaver()

agent = graph.compile(checkpointer=checkpointer)
