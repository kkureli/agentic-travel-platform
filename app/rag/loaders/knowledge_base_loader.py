from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)


def load_knowledge_base():
    loader = DirectoryLoader(
        "data/raw/knowledge_base",
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    return loader.load()


def enrich_metadata(documents):
    for document in documents:
        source = document.metadata["source"]

        if "policy" in source:
            document.metadata["document_type"] = "policy"

        if "hotel" in source:
            document.metadata["category"] = "hotel"

    return documents
