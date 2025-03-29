import chromadb
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import os


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

        pdf_collection.add(
            documents=chunks,
            metadatas=metadata,
            ids=[str(uuid.uuid4()) for _ in chunks]
        )


folder_path = "uploads"
pdf_files = [
    os.path.join(folder_path, file)
    for file in os.listdir(folder_path)
    if file.endswith(".pdf") or file.endswith(".doc") or file.endswith(".docx")
]

handle_multiple_pdfs(pdf_files)
