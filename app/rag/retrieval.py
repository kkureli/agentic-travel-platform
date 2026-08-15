from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.vector_store.qdrant_store import QdrantVectorStore


def retrieve(query: str, limit: int = 3):
    embedding_service = SentenceTransformerEmbeddingService()
    vector_store = QdrantVectorStore()

    query_vector = embedding_service.embed_text(query)

    return vector_store.search(
        query_vector=query_vector,
        limit=limit,
    )


if __name__ == "__main__":
    results = retrieve("When can I cancel my hotel without a penalty?")

    for result in results:
        print("\nScore:", result.score)
        print("Text:", result.payload["text"])
        print("Metadata:", result.payload["metadata"])
