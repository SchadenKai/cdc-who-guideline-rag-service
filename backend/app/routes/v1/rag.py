from typing import Annotated, Optional

from fastapi import APIRouter, Depends, UploadFile

# INDEXING AGENT deps
from app.routes.dependencies.rag import get_indexing_service, get_retrieval_service
from app.services.rag import IndexingService, RetrievalService
from app.utils import get_request_id

rag_router = APIRouter(prefix="/rag", tags=["rag"])


@rag_router.post("/ingest")
def ingest_website(
    website_url: str,
    indexing_service: Annotated[IndexingService, Depends(get_indexing_service)],
    request_id: Annotated[str, Depends(get_request_id)],
):
    return indexing_service.ingest_website(
        website_url=website_url, request_id=request_id
    )


@rag_router.post("/ingest/file")
def ingest_document(
    file_key: str,
    indexing_service: Annotated[IndexingService, Depends(get_indexing_service)],
    request_id: Annotated[str, Depends(get_request_id)],
):
    return indexing_service.ingest_document(file_key=file_key, request_id=request_id)


@rag_router.post("/upload")
def upload_file_route(
    file: Optional[UploadFile],
    indexing_service: Annotated[IndexingService, Depends(get_indexing_service)],
):
    return indexing_service.upload_file(pdf_file=file.file, filename=file.filename)


@rag_router.get("/objects/all")
def get_object_list(
    indexing_service: Annotated[IndexingService, Depends(get_indexing_service)],
    file_name: Optional[str] = None,
) -> list[str]:
    return indexing_service.get_object_list(file_name)


@rag_router.post("/extract")
def extract_md_content_from_file(
    indexing_service: Annotated[IndexingService, Depends(get_indexing_service)],
    file_key: str,
):
    return indexing_service.extract_md_content(file_key)


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
