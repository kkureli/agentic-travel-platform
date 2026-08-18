from langchain_core.tools import tool

from app.rag.retrieval.qdrant_hybrid_retriever import (
    QdrantHybridRetriever,
)


def build_travel_rag_tool(
    retriever: QdrantHybridRetriever,
):
    @tool
    def retrieve_travel_knowledge(
        query: str,
    ) -> str:
        """
        Search the private travel-platform knowledge base.

        Use this tool for questions about:
        - hotel policies
        - cancellations
        - refunds
        - insurance
        - loyalty
        - check-in
        - payments
        - reservation rules
        """

        results = retriever.search(
            query=query,
            tenant_id="travel-platform",
            top_k=5,
        )

        if not results:
            return "No relevant travel-platform knowledge was found."

        chunks = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            document = result.document

            source = document.metadata.get(
                "source",
                "unknown",
            )

            chunks.append(
                f"""
Result {index}
Source: {source}

{document.page_content}
""".strip()
            )

        return "\n\n---\n\n".join(chunks)

    return retrieve_travel_knowledge
