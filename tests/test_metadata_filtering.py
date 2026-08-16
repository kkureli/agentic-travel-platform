from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.loaders.langchain_loader import (
    load_documents,
)
from tests.eval_chunking_retrieval import retrieve

documents = load_documents()

embedding_service = SentenceTransformerEmbeddingService()

results = retrieve(
    query="What protection is available?",
    chunks=documents,
    embedding_service=embedding_service,
    top_k=3,
    metadata_filter={
        "document_type": "insurance",
    },
)

for result in results:
    print(
        result.metadata,
        result.page_content[:100],
    )
