from pathlib import Path

from app.rag.loaders.markdown_loader import load_markdown_directory


def test_load_markdown_directory():
    documents = load_markdown_directory(Path("data/raw/knowledge_base"))

    assert len(documents) == 5

    for document in documents:
        assert document.text
        assert document.metadata["source"]
