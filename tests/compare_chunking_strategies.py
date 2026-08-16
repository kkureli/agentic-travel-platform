from pathlib import Path
from statistics import mean

from langchain_core.documents import Document

from app.rag.chunking.recursive import split_documents
from app.rag.chunking.semantic import SemanticChunker
from app.rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingService


def print_stats(
    strategy_name: str,
    chunks: list[Document],
) -> None:
    sizes = [len(chunk.page_content) for chunk in chunks]

    print(f"\n{'=' * 60}")
    print(strategy_name)
    print(f"{'=' * 60}")

    print(f"Chunk count: {len(chunks)}")
    print(f"Average size: {mean(sizes):.1f}")
    print(f"Min size: {min(sizes)}")
    print(f"Max size: {max(sizes)}")

    for index, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {index + 1} ({len(chunk.page_content)} chars) ---")
        print(chunk.page_content)


text = Path("data/raw/semantic_test.md").read_text(encoding="utf-8")

document = Document(
    page_content=text,
    metadata={
        "source": "semantic_test.md",
    },
)

recursive_chunks = split_documents(
    [document],
    chunk_size=500,
    chunk_overlap=100,
)

semantic_chunker = SemanticChunker(
    embedding_service=SentenceTransformerEmbeddingService(),
    minimum_similarity=0.30,
    max_chunk_size=500,
    chunk_overlap=100,
)

semantic_chunks = semantic_chunker.split_document(document)

print_stats(
    "RECURSIVE CHUNKING",
    recursive_chunks,
)

print_stats(
    "SEMANTIC CHUNKING",
    semantic_chunks,
)
