from app.rag.loaders.knowledge_base_loader import load_knowledge_base
from app.rag.chunking.recursive import split_documents


documents = load_knowledge_base()

chunks = split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100,
)

print("Documents:", len(documents))
print("Chunks:", len(chunks))

for index, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {index + 1} ---")
    print(chunk.page_content)
    print(chunk.metadata)
