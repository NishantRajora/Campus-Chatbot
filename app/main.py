"""
FastAPI backend for CourseRAG.

Endpoints:
  POST /api/upload   -> PDF in, knowledge base rebuilt
  POST /api/ask       -> question in, grounded answer + sources out
  GET  /api/status    -> whether a knowledge base currently exists
  GET  /              -> serves the single-page UI
"""
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import UPLOADS_DIR, TOP_K
from app.pdf_processor import extract_pages
from app.chunker import chunk_pages
from app.embeddings import embed_texts
from app.vector_store import store
from app.rag import answer_question

app = FastAPI(title="CourseRAG")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/")
def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
def status():
    return {
        "ready": store.is_ready(),
        "num_chunks": len(store.chunks) if store.is_ready() else 0,
    }


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save to a unique path so concurrent uploads never collide
    dest_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest_path = UPLOADS_DIR / dest_name
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        pages = extract_pages(str(dest_path))
        if not pages:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in this PDF (it may be scanned/image-only).",
            )

        chunks = chunk_pages(pages, source=file.filename)
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        store.build(embeddings, chunks)
    finally:
        # Uploaded PDF isn't needed after extraction - only the index + chunks persist
        dest_path.unlink(missing_ok=True)

    return {
        "status": "ok",
        "filename": file.filename,
        "pages_processed": len(pages),
        "chunks_created": len(chunks),
    }


@app.post("/api/ask")
def ask(payload: AskRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(payload.question.strip(), top_k=payload.top_k or TOP_K)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama API error: {e}")

    return JSONResponse(result)
