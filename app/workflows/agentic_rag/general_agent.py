from app.workflows.agentic_rag.llm import get_chat_model
from app.workflows.agentic_rag.state import (
    AgenticRAGState,
)

llm = get_chat_model()


def general_agent(
    state: AgenticRAGState,
) -> dict:
    response = llm.invoke(
        [
            (
                "system",
                (
                    "You are the general assistant of a "
                    "travel platform. Answer casual or "
                    "general questions directly. "
                    "Do not invent platform-specific "
                    "policies."
                ),
            ),
            (
                "human",
                state["query"],
            ),
        ]
    )

    return {
        "answer": response.content,
    }
