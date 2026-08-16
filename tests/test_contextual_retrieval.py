from app.rag.chunking.recursive import (
    split_documents,
)
from app.rag.contextualizer import (
    DocumentContextualizer,
)
from app.rag.loaders.langchain_loader import (
    load_documents,
)
from app.services.llm_service import (
    LLMService,
)


def main() -> None:
    documents = load_documents()

    document = documents[0]

    chunks = split_documents(
        [document],
        chunk_size=300,
        chunk_overlap=50,
    )

    chunk = chunks[0]

    llm_service = LLMService()

    contextualizer = DocumentContextualizer(
        llm_service=llm_service,
    )

    contextualized_chunk = contextualizer.contextualize_chunk(
        document=document,
        chunk=chunk,
    )

    print("\nORIGINAL CHUNK:")
    print(chunk.page_content)

    print("\nGENERATED CONTEXT:")
    print(contextualized_chunk.metadata["retrieval_context"])

    print("\nCONTEXTUALIZED CHUNK:")
    print(contextualized_chunk.page_content)


if __name__ == "__main__":
    main()
