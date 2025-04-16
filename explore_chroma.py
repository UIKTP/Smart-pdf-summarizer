from chromadb import PersistentClient

# Connect to the same persistent DB
chroma_client = PersistentClient(path="./chroma_db")

# Optional: list all collections
collections = chroma_client.list_collections()
print("✅ Collections found:", [c.name for c in collections])

# Load your stored collection
collection = chroma_client.get_collection("pdf_docs")

# Get all documents and metadata
results = collection.get(include=["documents", "metadatas"])

print(f"\n📄 Found {len(results['documents'])} document chunks in ChromaDB.\n")

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"--- Chunk {i+1} ---")
    print(f"Text: {doc[:300]}...")  # first 300 characters
    print(f"Metadata: {meta}")
    print("-" * 80)
