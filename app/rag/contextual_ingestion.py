from langchain_core.documents import Document

from app.rag.chunking.recursive import split_documents
from app.rag.contextualizer import DocumentContextualizer
from app.rag.loaders.langchain_loader import load_documents
from app.services.llm_service import LLMService


def build_contextual_chunks(
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    documents = load_documents()

    llm_service = LLMService()

    contextualizer = DocumentContextualizer(
        llm_service=llm_service,
    )

    contextual_chunks: list[Document] = []

    for document in documents:
        chunks = split_documents(
            [document],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk in chunks:
            contextual_chunk = contextualizer.contextualize_chunk(
                document=document,
                chunk=chunk,
            )

            contextual_chunks.append(contextual_chunk)

    return contextual_chunks
