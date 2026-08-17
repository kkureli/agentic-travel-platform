from langchain_core.documents import Document

from app.rag.chunking.recursive import split_documents
from app.rag.chunking.semantic import SemanticChunker
from app.rag.embeddings.embedding_service import EmbeddingService


class RetrievalDocumentBuilder:
    def __init__(
        self,
        minimum_similarity: float = 0.30,
        max_chunk_size: int = 800,
        chunk_overlap: int = 100,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        self.semantic_chunker = SemanticChunker(
            embedding_service=embedding_service or EmbeddingService(),
            minimum_similarity=minimum_similarity,
        )

    def build(
        self,
        documents: list[Document],
    ) -> list[Document]:
        final_chunks: list[Document] = []

        for document in documents:
            semantic_chunks = self.semantic_chunker.split_document(document)

            for chunk in semantic_chunks:
                if len(chunk.page_content) > self.max_chunk_size:
                    recursive_chunks = split_documents(
                        [chunk],
                        chunk_size=self.max_chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )

                    final_chunks.extend(recursive_chunks)

                else:
                    final_chunks.append(chunk)

        return final_chunks
