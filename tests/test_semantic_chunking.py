from pathlib import Path

from langchain_core.documents import Document

from app.rag.chunking.semantic import SemanticChunker
from app.rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingService

if __name__ == "__main__":
    text = Path("data/raw/semantic_test.md").read_text(encoding="utf-8")

    document = Document(
        page_content=text,
        metadata={
            "source": "semantic_test.md",
        },
    )

    chunker = SemanticChunker(
        embedding_service=SentenceTransformerEmbeddingService(),
        minimum_similarity=0.30,
    )

    chunks = chunker.split_document(document)

    for index, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {index + 1} ---")
        print(chunk.page_content)
