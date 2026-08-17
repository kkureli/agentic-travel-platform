# Text
# ↓
# Tokenizer
# ↓
# Token IDs
# ↓
# Embedding Layer
# ↓
# Initial token vectors
# ↓
# Transformer
# ↓
# Context-aware token representations
# ↓
# Pooling
# ↓
# Final text embedding

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text)

        return embedding.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)

        return embeddings.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)
