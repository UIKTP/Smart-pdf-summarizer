import uuid
from chromadb import PersistentClient

# ✅ Persistent ChromaDB client
chroma_client = PersistentClient(path="./chroma_db")

# ✅ Get or create the collection
pdf_collection = chroma_client.get_or_create_collection(name="pdf_docs")

def store_chunks_in_chroma(chunks, metadata):
    ids = [str(uuid.uuid4()) for _ in chunks]
    pdf_collection.add(
        documents=chunks,
        metadatas=metadata,
        ids=ids
    )
    # No need to call .persist() anymore
    return len(ids)
