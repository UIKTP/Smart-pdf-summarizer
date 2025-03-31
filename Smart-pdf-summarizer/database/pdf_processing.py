import chromadb
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import os
import chromadb.utils.embedding_functions as embedding_functions


huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="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",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chroma_client = chromadb.Client()
pdf_collection = chroma_client.get_or_create_collection(name="pdf_docs")

def extract_text_from_pdf(pdf_file):
    pages = []
    reader = pypdf.PdfReader(pdf_file)

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        page_text = page_text.replace('\n', ' ').replace("  ", " ")
        if page_text:
            pages.append((page_text, page_num + 1))

    return pages

def handle_multiple_pdfs(pdf_files):
    for pdf_file in pdf_files:
        pdf_pages = extract_text_from_pdf(pdf_file)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        chunks = []
        metadata = []

        for page_text, page_number in pdf_pages:
            page_chunks = splitter.split_text(page_text)

            chunks.extend(page_chunks)
            metadata.extend([{"source": pdf_file, "page": page_number}] * len(
                page_chunks))

        embeddings = huggingface_ef(chunks)

        pdf_collection.add(
            documents=chunks,
            metadatas=metadata,
            embeddings=embeddings,
            ids=[str(uuid.uuid4()) for _ in chunks]
        )


folder_path = "uploads"
pdf_files = [
    os.path.join(folder_path, file)
    for file in os.listdir(folder_path)
    if file.endswith(".pdf") or file.endswith(".doc") or file.endswith(".docx")
]

handle_multiple_pdfs(pdf_files)

''' 
--TEST--
query_text = "What is artificial intelligence?"
query_embedding = huggingface_ef([query_text])

results = pdf_collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print(results)
'''