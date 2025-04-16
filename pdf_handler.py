import os
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_file):
    pages = []
    reader = pypdf.PdfReader(pdf_file)

    if not reader.pages:
        print(f"[!] No readable pages in {pdf_file}")
        return []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            page_text = page_text.replace('\n', ' ').replace("  ", " ")
            pages.append((page_text, page_num + 1))

    return pages


def split_pdf_to_chunks(pdf_file):
    pdf_pages = extract_text_from_pdf(pdf_file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    chunks = []
    metadata = []

    for page_text, page_number in pdf_pages:
        page_chunks = splitter.split_text(page_text)
        chunks.extend(page_chunks)
        metadata.extend([{"source": pdf_file, "page": page_number}] * len(page_chunks))

    return chunks, metadata
