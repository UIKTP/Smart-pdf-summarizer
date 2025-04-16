import pdfplumber
from langchain.text_splitter import CharacterTextSplitter

def load_pdf_text_and_tables(file_path):
    """Extract text and tables from a PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num} ---\n{page_text}"

            # Extract tables
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables, start=1):
                table_text = "\n".join(
                    [" | ".join(str(cell) if cell is not None else "" for cell in row) for row in table if any(row)]
                )
                text += f"\n--- Table {table_idx} (Page {page_num}) ---\n{table_text}"
    return text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.create_documents([text])
