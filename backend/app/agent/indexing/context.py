from langchain_core.embeddings import Embeddings
from langchain_text_splitters import TextSplitter
from pydantic import BaseModel, ConfigDict
from pymilvus import MilvusClient


class AgentContext(BaseModel):
    chunker: TextSplitter
    encoder: Embeddings
    db_client: MilvusClient
    collection_name: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
