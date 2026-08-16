from app.rag.chunking.recursive import (
    split_documents,
)
from app.rag.loaders.langchain_loader import (
    load_documents,
)
from app.rag.retrieval.bm25_retriever import (
    BM25Retriever,
)

documents = load_documents()

chunks = split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100,
)

retriever = BM25Retriever(
    documents=chunks,
)

queries = [
    "non-refundable reservation",
    "Belek family hotels",
    "loyalty program points",
]

for query in queries:
    print(f"\nQUERY: {query}")

    results = retriever.search(
        query=query,
        top_k=3,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            rank,
            result.metadata.get("source"),
            result.page_content[:100],
        )
