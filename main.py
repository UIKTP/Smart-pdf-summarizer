import os
import uuid
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import RedirectResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from pdf_handler import split_pdf_to_chunks
from chroma_manager import store_chunks_in_chroma
from docx import Document as DocxDocument
from datetime import datetime
import chromadb
import re
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi import Form

# Settings
FOLDER_PATH = "uploads"
CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "pdf_docs"
OUTPUT_DIR = "outputs"
CHAT_HISTORY_DIR = "chat_history"

# Ensure directories exist
os.makedirs(FOLDER_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="PDF Processing API")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

MAX_PDF_UPLOAD = 2
MAX_PDF_SIZE_MB = 1

# Pydantic models
class PDFItem(BaseModel):
    id: str
    name: str


class ChatItem(BaseModel):
    question: str
    answer: str
    timestamp: str


class DocumentResponse(BaseModel):
    id: str
    name: str
    chat_history: List[ChatItem]


# Helper functions
def simple_summarize(chunks: List[str], length: str = "medium") -> str:
    """
    Generate a simple summary by extracting key sentences based on length.
    - length: "short" (2-3 sentences), "medium" (4-6 sentences), "long" (7-10 sentences)
    """
    # Combine all chunks into one text
    text = " ".join(chunks)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Determine number of sentences based on length
    length_map = {"short": 3, "medium": 6, "long": 10}
    num_sentences = length_map.get(length, 6)

    # Simple heuristic: take first N sentences (could be improved with TF-IDF or sentence scoring)
    selected_sentences = sentences[:min(num_sentences, len(sentences))]
    return " ".join(selected_sentences)


def generate_test(chunks: List[str]) -> str:
    """
    Generate a simple test with 5 multiple-choice questions based on chunk content.
    """
    text = " ".join(chunks)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Generate 5 questions by selecting sentences and creating dummy options
    questions = []
    for i, sentence in enumerate(sentences[:5], 1):
        # Simple question: turn sentence into a question
        question_text = f"Question {i}: What is true about the following? {sentence}"
        options = [
            f"A) {sentence} (Correct)",
            f"B) Not {sentence}",
            "C) Unrelated option",
            "D) Another unrelated option"
        ]
        questions.append(f"{question_text}\n" + "\n".join(options))

    return "\n\n".join(questions)


def simple_qa(query: str, chunks: List[str]) -> str:
    """
    Perform a simple keyword-based search to answer a query.
    """
    query_words = set(query.lower().split())
    best_match = None
    max_overlap = 0

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(query_words.intersection(chunk_words))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = chunk

    if best_match:
        return best_match[:500] + "..." if len(best_match) > 500 else best_match
    return "No relevant information found in the PDF."


def save_to_docx(content: str, filename: str) -> str:
    doc = DocxDocument()
    doc.add_paragraph(content)
    filepath = os.path.join(OUTPUT_DIR, filename)
    doc.save(filepath)
    return filepath


def save_chat_history(doc_id: str, question: str, answer: str):
    history_file = os.path.join(CHAT_HISTORY_DIR, f"{doc_id}.json")
    history = []
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    history.append({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    })
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def get_chat_history(doc_id: str) -> List[ChatItem]:
    history_file = os.path.join(CHAT_HISTORY_DIR, f"{doc_id}.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            return [ChatItem(**item) for item in json.load(f)]
    return []


# Endpoints
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    try:
        pdf_files = [
            {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f)), "name": f}
            for f in os.listdir(FOLDER_PATH)
            if f.lower().endswith(".pdf")
        ]
        return templates.TemplateResponse("home.html", {"request": request, "pdfs": pdf_files})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing PDFs: {str(e)}")



@app.get("/document/{id}",response_class=HTMLResponse)
async def get_document(id: str,request: Request):
    try:
        pdf_files = {str(uuid.uuid5(uuid.NAMESPACE_DNS, f)): f for f in os.listdir(FOLDER_PATH) if
                     f.lower().endswith(".pdf")}
        if id not in pdf_files:
            raise HTTPException(status_code=404, detail="Document not found")

        name = pdf_files[id]
        chat_history = get_chat_history(id)
        return templates.TemplateResponse("document.html", {
            "request": request,
            "id": id,
            "name": name,
            "chat_history": chat_history
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@app.post("/document/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        pdf_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(".pdf")]
        if len(pdf_files)+1 > MAX_PDF_UPLOAD:
            raise HTTPException(
                status_code=400, 
                detail=f"You cannot upload more than {MAX_PDF_UPLOAD} PDF documents."
            )
        
        content = await file.read()

        size_mb = len(content) / (1024 * 1024) 
        if size_mb > MAX_PDF_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"The size of the PDF document must not exceed {MAX_PDF_SIZE_MB} MB."
            )

        file_path = os.path.join(FOLDER_PATH, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)

        # Process the uploaded PDF
        chunks, metadata = split_pdf_to_chunks(file_path)
        if chunks:
            store_chunks_in_chroma(chunks, metadata)

        return RedirectResponse(url="/home", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@app.get("/document/summarizer/{id}")
async def summarize_document(id: str):
    try:
        pdf_files = {str(uuid.uuid5(uuid.NAMESPACE_DNS, f)): f for f in os.listdir(FOLDER_PATH) if
                     f.lower().endswith(".pdf")}
        if id not in pdf_files:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = os.path.join(FOLDER_PATH, pdf_files[id])
        chunks, _ = split_pdf_to_chunks(file_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in PDF")

        summary = simple_summarize(chunks, length="medium")
        filename = f"summary_{pdf_files[id]}.docx"
        file_path = save_to_docx(summary, filename)

        return FileResponse(
            path=file_path,
            filename=filename,
            headers={"X-Redirect": f"/document/{id}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@app.get("/document/test/{id}")
async def generate_test_document(id: str):
    try:
        pdf_files = {str(uuid.uuid5(uuid.NAMESPACE_DNS, f)): f for f in os.listdir(FOLDER_PATH) if
                     f.lower().endswith(".pdf")}
        if id not in pdf_files:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = os.path.join(FOLDER_PATH, pdf_files[id])
        chunks, _ = split_pdf_to_chunks(file_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in PDF")

        test = generate_test(chunks)
        filename = f"test_{pdf_files[id]}.docx"
        file_path = save_to_docx(test, filename)

        return FileResponse(
            path=file_path,
            filename=filename,
            headers={"X-Redirect": f"/document/{id}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating test: {str(e)}")


@app.post("/document/{id}")
async def ask_question(id: str, question: str = Form(...)):
    try:
        pdf_files = {str(uuid.uuid5(uuid.NAMESPACE_DNS, f)): f for f in os.listdir(FOLDER_PATH) if
                     f.lower().endswith(".pdf")}
        if id not in pdf_files:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = os.path.join(FOLDER_PATH, pdf_files[id])
        chunks, _ = split_pdf_to_chunks(file_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in PDF")

        answer = simple_qa(question, chunks)
        save_chat_history(id, question, answer)

        return RedirectResponse(url=f"/document/{id}", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/document/delete/{id}")
async def delete_document(id: str, request: Request):
    try:
        
        pdf_files = {str(uuid.uuid5(uuid.NAMESPACE_DNS, f)): f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(".pdf")}
        
        if id not in pdf_files:
            raise HTTPException(status_code=404, detail="Document not found")

        filename = pdf_files[id]
        file_path = os.path.join(FOLDER_PATH, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)

        history_path = os.path.join(CHAT_HISTORY_DIR, f"{id}.json")
        if os.path.exists(history_path):
            os.remove(history_path)

        for suffix in ["summary_", "test_"]:
            docx_file = os.path.join(OUTPUT_DIR, f"{suffix}{filename.replace('.pdf', '.docx')}")
            if os.path.exists(docx_file):
                os.remove(docx_file)

        return RedirectResponse(url="/home", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)