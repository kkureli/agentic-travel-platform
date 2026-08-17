from langchain_qdrant import FastEmbedSparse


class BM25SparseEmbeddingService:
    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
    ):
        self.model = FastEmbedSparse(
            model_name=model_name,
        )

    def embed_documents(
        self,
        texts: list[str],
    ):
        return self.model.embed_documents(texts)

    def embed_query(
        self,
        text: str,
    ):
        return self.model.embed_query(text)
