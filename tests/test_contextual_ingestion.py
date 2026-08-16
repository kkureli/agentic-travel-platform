from app.rag.contextual_ingestion import (
    build_contextual_chunks,
)


def main() -> None:
    chunks = build_contextual_chunks(
        chunk_size=500,
        chunk_overlap=100,
    )

    print(f"\nContextual chunk count: {len(chunks)}")

    for index, chunk in enumerate(
        chunks[:3],
        start=1,
    ):
        print(f"\n{'=' * 60}")

        print(f"CHUNK {index}")

        print(f"{'=' * 60}")

        print("\nSOURCE:")

        print(chunk.metadata.get("source"))

        print("\nCONTEXT:")

        print(chunk.metadata.get("retrieval_context"))

        print("\nFINAL TEXT:")

        print(chunk.page_content)


if __name__ == "__main__":
    main()
