from langchain_core.documents import Document
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from app.services.llm_service import LLMService


class DocumentContextualizer:
    def __init__(
        self,
        llm_service: LLMService,
    ):
        self.llm_service = llm_service

    def contextualize_chunk(
        self,
        document: Document,
        chunk: Document,
    ) -> Document:
        messages = [
            SystemMessage(
                content=(
                    "You generate short context for retrieval. "
                    "Given a full document and one chunk from that document, "
                    "write a short context that explains where the chunk belongs "
                    "and what it refers to. "
                    "Do not summarize the entire document. "
                    "Return only the short context."
                )
            ),
            HumanMessage(
                content=(
                    f"<document>\n"
                    f"{document.page_content}\n"
                    f"</document>\n\n"
                    f"<chunk>\n"
                    f"{chunk.page_content}\n"
                    f"</chunk>"
                )
            ),
        ]

        response = self.llm_service.generate(messages)

        context = response.content.strip()

        contextualized_text = f"{context}\n\n{chunk.page_content}"

        metadata = chunk.metadata.copy()
        metadata["contextualized"] = True
        metadata["retrieval_context"] = context

        return Document(
            page_content=contextualized_text,
            metadata=metadata,
        )
