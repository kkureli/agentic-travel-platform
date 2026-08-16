from langchain_core.documents import Document
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from app.rag.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from app.rag.retrieval.rrf import (
    reciprocal_rank_fusion,
)
from app.services.llm_service import (
    LLMService,
)


class MultiQueryRetriever:
    def __init__(
        self,
        llm_service: LLMService,
        retriever: HybridRetriever,
        query_count: int = 3,
    ):
        self.llm_service = llm_service
        self.retriever = retriever
        self.query_count = query_count

    def generate_queries(
        self,
        query: str,
    ) -> list[str]:
        messages = [
            SystemMessage(
                content=(
                    "Generate multiple search queries for document retrieval. "
                    "All queries must preserve the user's original intent. "
                    "Use different wording or perspectives to improve recall. "
                    f"Generate exactly {self.query_count} queries. "
                    "Return one query per line. "
                    "Do not number the queries. "
                    "Do not answer the question."
                )
            ),
            HumanMessage(content=query),
        ]

        response = self.llm_service.generate(messages)

        queries = [
            line.strip() for line in response.content.splitlines() if line.strip()
        ]

        return queries[: self.query_count]

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 8,
    ) -> list[Document]:
        generated_queries = self.generate_queries(query)

        rankings: list[list[Document]] = []

        for generated_query in generated_queries:
            results = self.retriever.search(
                query=generated_query,
                top_k=candidate_k,
                candidate_k=candidate_k,
            )

            rankings.append(results)

        return reciprocal_rank_fusion(
            rankings=rankings,
            top_k=top_k,
        )
