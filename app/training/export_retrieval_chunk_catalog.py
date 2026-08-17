import json
import uuid
from pathlib import Path

from app.rag.ingestion.retrieval_document_builder import (
    RetrievalDocumentBuilder,
)
from app.rag.loaders.langchain_loader import load_documents

OUTPUT_PATH = Path("data/training/retrieval_chunk_catalog.jsonl")

TENANT_ID = "travel-platform"


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def build_document_id(
    tenant_id: str,
    source: str,
) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{tenant_id}:{source}",
        )
    )


def build_chunk_id(
    document_id: str,
    text: str,
) -> str:
    normalized_text = normalize_text(text)

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{document_id}:{normalized_text}",
        )
    )


def main() -> None:
    documents = load_documents()

    builder = RetrievalDocumentBuilder()

    chunks = builder.build(documents)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in chunks:
            source = str(
                chunk.metadata.get(
                    "source",
                    "unknown",
                )
            )

            document_id = build_document_id(
                tenant_id=TENANT_ID,
                source=source,
            )

            chunk_id = build_chunk_id(
                document_id=document_id,
                text=chunk.page_content,
            )

            row = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source": source,
                "text": chunk.page_content,
            }

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Exported {len(chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
