import os
from pdf_handler import split_pdf_to_chunks
from chroma_manager import store_chunks_in_chroma

FOLDER_PATH = "uploads"

def process_all_pdfs():
    pdf_files = [
        os.path.join(FOLDER_PATH, f)
        for f in os.listdir(FOLDER_PATH)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("[!] No PDF files found in 'uploads' folder.")
        return

    for file in pdf_files:
        print(f"\n[+] Processing {file}")
        chunks, metadata = split_pdf_to_chunks(file)

        if chunks:
            total = store_chunks_in_chroma(chunks, metadata)
            print(f"[✓] Stored {total} chunks from {file}")
        else:
            print(f"[!] No text found in {file}")

if __name__ == "__main__":
    process_all_pdfs()
