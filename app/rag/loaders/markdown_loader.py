from pathlib import Path

from app.schemas.document import Document


def load_markdown(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")

    return Document(
        text=text,
        metadata={
            "source": path.name,
            "file_path": str(path),
            "file_type": "markdown",
        },
    )


def load_markdown_directory(directory: Path) -> list[Document]:
    documents: list[Document] = []

    for path in directory.glob("*.md"):
        document = load_markdown(path)
        documents.append(document)

    return documents
