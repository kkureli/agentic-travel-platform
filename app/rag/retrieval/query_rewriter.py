from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from app.services.llm_service import LLMService


class QueryRewriter:
    def __init__(
        self,
        llm_service: LLMService,
    ):
        self.llm_service = llm_service

    def rewrite(
        self,
        query: str,
    ) -> str:
        messages = [
            SystemMessage(
                content=(
                    "Rewrite the user's query for document retrieval. "
                    "Preserve the original intent. "
                    "Make the query explicit and searchable. "
                    "Do not answer the question. "
                    "Return only the rewritten query."
                )
            ),
            HumanMessage(content=query),
        ]

        response = self.llm_service.generate(messages)

        return response.content.strip()
