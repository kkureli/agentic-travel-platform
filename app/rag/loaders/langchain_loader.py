from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_core.documents import Document


def load_documents(
    path: str = "data/raw/knowledge_base",
) -> list[Document]:
    loader = DirectoryLoader(
        path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    for document in documents:
        source = document.metadata.get("source", "")
        filename = Path(source).name

        document.metadata["document_type"] = _infer_document_type(filename)

        document.metadata["language"] = "en"

    return documents


def _infer_document_type(
    filename: str,
) -> str:
    if "policy" in filename:
        return "policy"

    if "insurance" in filename:
        return "insurance"

    if "guide" in filename:
        return "guide"

    if "loyalty" in filename:
        return "loyalty"

    return "general"
