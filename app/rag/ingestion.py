from app.rag.chunking.recursive import split_documents
from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.loaders.knowledge_base_loader import (
    load_knowledge_base,
)
from app.rag.vector_store.qdrant_store import QdrantVectorStore


def ingest_knowledge_base() -> None:
    documents = load_knowledge_base()

    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    embedding_service = SentenceTransformerEmbeddingService()

    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedding_service.embed_texts(texts)

    vector_store = QdrantVectorStore()

    vector_store.create_collection()

    vector_store.index_documents(
        chunks,
        embeddings,
    )

    print(f"Indexed {len(chunks)} chunks.")


if __name__ == "__main__":
    ingest_knowledge_base()
