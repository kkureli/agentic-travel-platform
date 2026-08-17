from langchain_core.documents import Document
from sentence_transformers.util import cos_sim

from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)


class DenseRetriever:
    def __init__(
        self,
        documents: list[Document],
        embedding_service: SentenceTransformerEmbeddingService,
    ):
        self.documents = documents
        self.embedding_service = embedding_service

        self.document_embeddings = self.embedding_service.embed_documents(
            [document.page_content for document in documents]
        )

    def search_with_scores(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        query_embedding = self.embedding_service.embed_query(query)

        similarities = cos_sim(
            query_embedding,
            self.document_embeddings,
        )[0]

        ranked_indices = similarities.argsort(descending=True)[:top_k]

        results: list[tuple[Document, float]] = []

        for index in ranked_indices:
            document = self.documents[int(index)]

            score = float(similarities[index])

            results.append((document, score))

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Document]:
        results = self.search_with_scores(
            query=query,
            top_k=top_k,
        )

        return [document for document, _ in results]
