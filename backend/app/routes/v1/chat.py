from typing import Optional, cast
from fastapi import APIRouter, Depends

from app.services.llm.factory import get_llm_provider
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from app.agent.react_agent.main import agent
from app.agent.react_agent.state import AgentState
from langchain_core.messages import HumanMessage, SystemMessage

# from ag_ui.core import RunAgentInput
# from ag_ui.encoder import EventEncoder
from app.agent.react_agent.context import AgentContext

chat_router = APIRouter(prefix="/chat", tags=["chat"])


@chat_router.post("")
def send_message(
    query: str,
    system_prompt: str | None = "You are a helpful assistant",
    llm: BaseChatModel = Depends(get_llm_provider),
):
    init_state = AgentState(
        messages=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]
    )
    context = AgentContext(llm=llm)
    final_response = ""
    for res in agent.stream(input=init_state, context=context):
        for node_name, state_update in res.items():
            if "messages" in state_update:
                final_message = cast(BaseMessage, state_update["messages"][-1])
                final_response = final_message.content

    return final_response


# @chat_router.post("/ag-ui/stream")
# def send_message_agui(req: Request):
#     # used for encoding response
#     encoder = EventEncoder()
