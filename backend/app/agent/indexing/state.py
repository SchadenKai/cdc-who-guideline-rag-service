from typing import Optional

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict

from app.agent.indexing.models import ProgressStatusEnum, RelevantDocs


class AgentState(BaseModel):
    file_path: Optional[str] = None
    raw_document: Optional[list[Document]] = None
    chunked_documents: Optional[list[Document]] = None
    final_documents: Optional[list[RelevantDocs]] = None
    progress_status: Optional[ProgressStatusEnum] = None
    run_metadata: Optional[dict] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
