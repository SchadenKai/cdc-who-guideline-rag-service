from logging import Logger

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from app.agent.indexing.context import AgentContext as IndexingAgentContext
from app.agent.indexing.state import AgentState
from app.agent.indexing.state import AgentState as IndexingAgentState
from app.core.config import Settings
from app.rag.chunker import ChunkerService
from app.rag.db import VectorClient
from app.rag.embeddings import EmbeddingService
from app.services.llm.factory import ChatModelService
from app.services.llm.tokenizer import TokenizerService


class RAGService:
    def __init__(
        self,
        chunker_service: ChunkerService,
        encoder_service: EmbeddingService,
        vector_db_service: VectorClient,
        tokenizer_service: TokenizerService,
        chat_model_service: ChatModelService,
        indexing_agent: CompiledStateGraph,
        settings: Settings,
        app_logger: Logger,
    ):
        self.chunker_service: ChunkerService = chunker_service
        self.encoder_service: EmbeddingService = encoder_service
        self.vector_db_service: VectorClient = vector_db_service
        self.tokenizer_service: TokenizerService = tokenizer_service
        self.chat_model_service: ChatModelService = chat_model_service
        self.indexing_agent: CompiledStateGraph = indexing_agent
        self.settings: Settings = settings
        self.app_logger: Logger = app_logger

    def ingest_document(self, website_url: str, request_id: str) -> AgentState:
        collection_name = self.settings.milvus_collection_name
        db_client = self.vector_db_service.client

        init_state = IndexingAgentState(website_url=website_url)
        context = IndexingAgentContext(
            chunker=self.chunker_service.get("recursive"),
            embedding=self.encoder_service,
            tokenizer=self.tokenizer_service,
            db_client=db_client,
            collection_name=collection_name,
        )
        config: RunnableConfig = {"configurable": {"thread_id": request_id}}
        final_response = {}
        for res in self.indexing_agent.stream(
            input=init_state, context=context, config=config
        ):
            for node_name, state in res.items():
                if "indexing_node" in node_name:
                    final_response = state["run_metadata"]
        return final_response
    def retrieve_documents