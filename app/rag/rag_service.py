from langchain_core.messages import HumanMessage, SystemMessage

from app.rag.retrieval import retrieve
from app.schemas.rag import RAGResponse, Source
from app.services.llm_service import LLMService


class RAGService:
    def __init__(self):
        self.llm = LLMService()

    def ask(self, question: str) -> RAGResponse:
        results = retrieve(
            query=question,
            limit=3,
        )

        context = "\n\n".join(result.payload["text"] for result in results)

        messages = [
            SystemMessage(
                content=(
                    "You are a travel assistant. "
                    "Answer the user's question using only the provided context. "
                    "If the context does not contain enough information, "
                    "say that you do not know."
                )
            ),
            HumanMessage(
                content=f"""
Context:

{context}

Question:

{question}
"""
            ),
        ]

        response = self.llm.generate(messages)

        sources = [
            Source(
                source=result.payload["metadata"]["source"],
                score=result.score,
                text=result.payload["text"],
            )
            for result in results
        ]

        return RAGResponse(
            answer=response.content,
            sources=sources,
        )


if __name__ == "__main__":
    rag = RAGService()

    result = rag.ask("When can I cancel a standard hotel reservation without penalty?")

    print("Answer:")
    print(result.answer)

    print("\nSources:")

    for source in result.sources:
        print(
            source.source,
            source.score,
        )
