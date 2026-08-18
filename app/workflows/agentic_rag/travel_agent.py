from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool

from app.workflows.agentic_rag.llm import (
    get_chat_model,
)


def build_travel_agent(
    rag_tool: BaseTool,
):
    llm = get_chat_model()

    llm_with_tools = llm.bind_tools(
        [
            rag_tool,
        ]
    )

    def call_model(
        query: str,
    ):
        response = llm_with_tools.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a travel-platform assistant. "
                        "Use the available tools when you need "
                        "private travel-platform knowledge. "
                        "Do not invent platform policies."
                    )
                ),
                HumanMessage(content=query),
            ]
        )

        return response

    return call_model
