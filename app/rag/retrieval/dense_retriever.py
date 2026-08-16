from langchain_core.documents import Document
from sentence_transformers import util

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

        self.document_embeddings = self.embedding_service.model.encode(
            [document.page_content for document in documents],
            convert_to_tensor=True,
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[Document]:
        query_embedding = self.embedding_service.model.encode(
            query,
            convert_to_tensor=True,
        )

        similarities = util.cos_sim(
            query_embedding,
            self.document_embeddings,
        )[0]

        ranked_indices = similarities.argsort(descending=True)[:top_k]

        return [self.documents[index] for index in ranked_indices]
