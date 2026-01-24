from typing import Annotated

from fastapi import APIRouter, Depends

# from ag_ui.core import RunAgentInput
# from ag_ui.encoder import EventEncoder
from app.routes.dependencies.chat_service import get_chat_service
from app.services.chat import ChatService
from app.utils import get_request_id

chat_router = APIRouter(prefix="/chat", tags=["chat"])


@chat_router.post("")
def send_message(
    query: str,
    request_id: Annotated[str, Depends(get_request_id)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    system_prompt: str = "You are a helpful assistant",
):
    return chat_service.send_message(
        query=query, system_prompt=system_prompt, request_id=request_id
    )


# @chat_router.post("/ag-ui/stream")
# def send_message_agui(req: Request):
#     # used for encoding response
#     encoder = EventEncoder()
