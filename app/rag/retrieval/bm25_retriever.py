import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(
        self,
        documents: list[Document],
    ):
        self.documents = documents

        self.tokenized_corpus = [
            self._tokenize(document.page_content) for document in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[Document]:
        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]

        return [self.documents[index] for index in ranked_indices]

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )
