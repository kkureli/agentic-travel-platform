from langchain_core.documents import Document


def reciprocal_rank_fusion(
    rankings: list[list[Document]],
    k: int = 60,
    top_k: int = 3,
) -> list[Document]:
    scores: dict[str, float] = {}
    documents_by_key: dict[str, Document] = {}

    for ranking in rankings:
        for rank, document in enumerate(
            ranking,
            start=1,
        ):
            key = _document_key(document)

            documents_by_key[key] = document

            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    sorted_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [documents_by_key[key] for key in sorted_keys[:top_k]]


def _document_key(
    document: Document,
) -> str:
    source = document.metadata.get("source", "")

    return f"{source}:{document.page_content}"
