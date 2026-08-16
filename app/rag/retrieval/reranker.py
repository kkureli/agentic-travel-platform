from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 3,
    ) -> list[Document]:
        if not documents:
            return []

        pairs = [
            (
                query,
                document.page_content,
            )
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(
                documents,
                scores,
                strict=True,
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        return [document for document, _ in ranked[:top_k]]
