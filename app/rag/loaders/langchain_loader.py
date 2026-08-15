from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)

loader = DirectoryLoader(
    "data/raw/knowledge_base",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)

documents = loader.load()
