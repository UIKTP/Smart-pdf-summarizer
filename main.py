import os
from pdf_handler import split_pdf_to_chunks
from chroma_manager import store_chunks_in_chroma
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

FOLDER_PATH = "uploads"

# Ensure uploads folder exists
os.makedirs(FOLDER_PATH, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="PDF Processing API")


# Original function, unchanged
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


# Pydantic model for POST response
class ProcessResponse(BaseModel):
    message: str
    processed_files: List[Dict[str, str | int]]


# Endpoint 1: POST /process - Trigger PDF processing
@app.post("/process", response_model=ProcessResponse)
async def process_pdfs():
    try:
        pdf_files = [
            os.path.join(FOLDER_PATH, f)
            for f in os.listdir(FOLDER_PATH)
            if f.lower().endswith(".pdf")
        ]

        if not pdf_files:
            return ProcessResponse(
                message="No PDF files found in 'uploads' folder.",
                processed_files=[]
            )

        results = []
        for file in pdf_files:
            chunks, metadata = split_pdf_to_chunks(file)
            if chunks:
                total = store_chunks_in_chroma(chunks, metadata)
                results.append({"file": file, "chunks_stored": total})
            else:
                results.append({"file": file, "chunks_stored": 0, "error": "No text found"})

        return ProcessResponse(
            message="PDF processing completed.",
            processed_files=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")


# Endpoint 2: GET /files - List PDFs in uploads folder
@app.get("/files")
async def list_pdf_files():
    try:
        pdf_files = [
            f for f in os.listdir(FOLDER_PATH)
            if f.lower().endswith(".pdf")
        ]
        return {
            "message": f"Found {len(pdf_files)} PDF files.",
            "files": pdf_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


# Endpoint 3: GET /chunks - Retrieve stored chunks metadata
@app.get("/chunks")
async def get_chunks():
    try:
        from chromadb import PersistentClient
        chroma_client = PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection("pdf_docs")
        results = collection.get(include=["metadatas"])

        chunk_count = len(results["metadatas"])
        return {
            "message": f"Found {chunk_count} chunks in ChromaDB.",
            "chunk_count": chunk_count,
            "metadata": results["metadatas"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")


# Original main block, unchanged
if __name__ == "__main__":
    process_all_pdfs()
    # Optionally start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)