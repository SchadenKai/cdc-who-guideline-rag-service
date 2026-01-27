from typing import Annotated

from fastapi import APIRouter, Depends

# INDEXING AGENT deps
from app.routes.dependencies.rag import get_indexing_service, get_retrieval_service
from app.services.rag import IndexingService, RetrievalService
from app.utils import get_request_id

rag_router = APIRouter(prefix="/rag", tags=["rag"])


@rag_router.post("/ingest")
def ingest_document(
    website_url: str,
    indexing_service: Annotated[IndexingService, Depends(get_indexing_service)],
    request_id: Annotated[str, Depends(get_request_id)],
):
    return indexing_service.ingest_document(
        website_url=website_url, pdf_file=pdf_file, request_id=request_id
    )


@rag_router.post("/upload")
def upload_file_route(
    file: Optional[UploadFile],
    indexing_service: Annotated[IndexingService, Depends(get_indexing_service)],
    s3_service: Annotated[S3Service, Depends(get_s3_service)],
):
    return indexing_service.upload_file(
        pdf_file=file.file, s3_service=s3_service, filename=file.filename
    )


@rag_router.post("/retrieve")
def retrieve_documents(
    query: str,
    retriever_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    request_id: Annotated[str, Depends(get_request_id)],
    is_llm_enabled: bool = False,
):
    return retriever_service.retrieve_documents(
        query=query,
        is_llm_enabled=is_llm_enabled,
        request_id=request_id,
    )
