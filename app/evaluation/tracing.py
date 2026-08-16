from langsmith import traceable


@traceable(name="hybrid_retrieval")
def trace_hybrid_retrieval(
    retriever,
    query: str,
    top_k: int = 5,
):
    return retriever.search(
        query=query,
        top_k=top_k,
    )


@traceable(name="reranking")
def trace_reranking(
    reranker,
    query: str,
    documents,
    top_k: int = 3,
):
    return reranker.rerank(
        query=query,
        documents=documents,
        top_k=top_k,
    )
