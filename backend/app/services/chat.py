from typing import cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from app.agent.react_agent.context import AgentContext
from app.agent.react_agent.state import AgentState
from app.services.llm.factory import ChatModelService


class ChatService:
    def __init__(
        self,
        chat_model_service: ChatModelService,
        agent: CompiledStateGraph,
    ):
        self.chat_model_service = chat_model_service
        self.agent = agent

    def send_message(
        self,
        query: str,
        request_id: str,
        system_prompt: str = "You are a helpful assistant",
    ) -> AgentState:
        chat_model = self.chat_model_service.client
        init_state = AgentState(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ]
        )
        context = AgentContext(llm=chat_model)
        config: RunnableConfig = {"configurable": {"thread_id": request_id}}
        final_response = ""
        for res in self.agent.stream(input=init_state, context=context, config=config):
            for _, state_update in res.items():
                if "messages" in state_update:
                    final_message = cast(BaseMessage, state_update["messages"][-1])
                    final_response = final_message.content

        return final_response
