import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import util

from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)


class SemanticChunker:
    def __init__(
        self,
        embedding_service: SentenceTransformerEmbeddingService,
        minimum_similarity: float = 0.30,
        max_chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.embedding_service = embedding_service
        self.minimum_similarity = minimum_similarity
        self.max_chunk_size = max_chunk_size

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_document(
        self,
        document: Document,
    ) -> list[Document]:
        sentences = self._split_sentences(document.page_content)

        if not sentences:
            return []

        if len(sentences) == 1:
            return self._apply_recursive_fallback(document)

        embeddings = self.embedding_service.model.encode(
            sentences,
            convert_to_tensor=True,
        )

        semantic_chunks: list[str] = []
        current_chunk = [sentences[0]]

        for index in range(len(sentences) - 1):
            current_embedding = embeddings[index]
            next_embedding = embeddings[index + 1]

            similarity = util.cos_sim(
                current_embedding,
                next_embedding,
            ).item()

            if similarity >= self.minimum_similarity:
                current_chunk.append(sentences[index + 1])
            else:
                semantic_chunks.append(" ".join(current_chunk))

                current_chunk = [sentences[index + 1]]

        if current_chunk:
            semantic_chunks.append(" ".join(current_chunk))

        final_chunks: list[Document] = []

        for semantic_chunk in semantic_chunks:
            semantic_document = Document(
                page_content=semantic_chunk,
                metadata=document.metadata.copy(),
            )

            final_chunks.extend(self._apply_recursive_fallback(semantic_document))

        return final_chunks

    def _apply_recursive_fallback(
        self,
        document: Document,
    ) -> list[Document]:
        if len(document.page_content) <= self.max_chunk_size:
            return [document]

        return self.recursive_splitter.split_documents([document])

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )

        return [sentence.strip() for sentence in sentences if sentence.strip()]
