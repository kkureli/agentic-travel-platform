from dotenv import load_dotenv

load_dotenv(".env", override=True)

from app.workflows.agentic_rag.graph import (
    build_agentic_rag_graph,
)


def main() -> None:
    graph = build_agentic_rag_graph()

    query = "What compensation do I receive if my hotel room has a bad sea view?"

    print(f"\nQUERY:\n{query}\n")

    for chunk in graph.stream(
        {
            "query": query,
            "retry_count": 0,
        },
        subgraphs=True,
        stream_mode="updates",
        version="v2",
    ):
        if chunk["type"] != "updates":
            continue

        namespace = chunk["ns"]
        data = chunk["data"]

        if namespace:
            print(f"\nSUBGRAPH: {namespace}")
        else:
            print("\nPARENT GRAPH")

        for node_name, update in data.items():
            print(f"NODE: {node_name}")

            if not update:
                continue

            if "route" in update:
                print(f"  route: {update['route']}")

            if "rewritten_query" in update:
                print(f"  rewritten_query: {update['rewritten_query']}")

            if "retry_count" in update:
                print(f"  retry_count: {update['retry_count']}")

            if "evidence_sufficient" in update:
                print(f"  evidence_sufficient: {update['evidence_sufficient']}")

            if "documents" in update:
                print(f"  documents: {len(update['documents'])}")

            if "answer" in update:
                print(f"  answer: {update['answer']}")


if __name__ == "__main__":
    main()
