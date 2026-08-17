from dotenv import load_dotenv

from app.workflows.agentic_rag.graph import build_agentic_rag_graph

load_dotenv(
    ".env",
    override=True,
)


def run_query(graph, query: str) -> None:

    result = graph.invoke(
        {
            "query": query,
            "retry_count": 0,
        }
    )

    print("\nQUERY:")
    print(query)

    print("\nROUTE:")
    print(result.get("route"))

    print("\nRETRY COUNT:")
    print(result.get("retry_count", 0))

    print("\nANSWER:")
    print(result.get("answer"))

    documents = result.get(
        "documents",
        [],
    )

    if documents:
        print("\nSOURCES:")

        for i, document in enumerate(
            documents,
            start=1,
        ):
            print(f"{i}. {document.metadata.get('source')}")


def main() -> None:
    graph = build_agentic_rag_graph()

    queries = ["What compensation do I receive if my hotel room has a bad sea view?"]

    for query in queries:
        print("\n" + "=" * 80)

        run_query(graph, query)


if __name__ == "__main__":
    main()
