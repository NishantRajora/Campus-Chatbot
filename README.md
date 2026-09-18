# CourseRAG

A local, strict RAG pipeline over an uploaded PDF:

```
PDF → PyMuPDF text extraction → chunk (800 / overlap 150)
    → all-MiniLM-L6-v2 embeddings → FAISS (local, index.faiss + chunks.pkl)
    → question embedding → FAISS top-5 search
    → strict prompt → Ollama Cloud (gpt-oss:20b) → grounded answer + sources
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste your real OLLAMA_API_KEY
```

`.env` (already in `.gitignore` — never commit it, and the key never appears
in any HTML/JS served to the browser; only the backend reads it):

```
OLLAMA_API_KEY=xxx
OLLAMA_API_URL=https://ollama.com
OLLAMA_MODEL=gpt-oss:20b

TOP_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000**.

1. Drag a PDF in (or choose one) → **Add to knowledge base**. This extracts
   text page-by-page, chunks it, embeds each chunk, and (re)builds
   `vector_store/index.faiss` + `vector_store/chunks.pkl`.
2. Ask a question → the backend embeds the question, retrieves the top
   `TOP_K` chunks from FAISS, and sends only those chunks — never the
   whole document — to Ollama Cloud inside the strict prompt.
3. The UI shows the answer plus the retrieved sources, each with its page
   number and cosine similarity.

Uploading a new PDF replaces the existing knowledge base (single-document
setup, matching the architecture above). Re-run the upload endpoint per
document if you want to swap documents.

## Project layout

```
courserag/
├── app/
│   ├── main.py            FastAPI app: /api/upload, /api/ask, /api/status
│   ├── config.py          loads .env (CHUNK_SIZE, TOP_K, Ollama settings)
│   ├── pdf_processor.py   PyMuPDF extraction, page-by-page
│   ├── chunker.py         800-char chunks, 150 overlap, per page
│   ├── embeddings.py      sentence-transformers/all-MiniLM-L6-v2
│   ├── vector_store.py    FAISS IndexFlatIP + pickled chunk metadata
│   └── rag.py             retrieval + strict prompt + Ollama Cloud call
├── static/
│   └── index.html         single-page UI (upload, ask, answer, sources)
├── vector_store/          index.faiss + chunks.pkl written here on upload
├── uploads/                temp storage for a PDF during processing only
├── requirements.txt
├── .env.example
└── .gitignore
```

## Notes on the strict prompt

The system prompt instructs the model to answer only from the retrieved
context, to say so explicitly when information is missing
(`"I could not find this information in the provided document."`), and to
flag partial answers rather than fill gaps from outside knowledge. Because
only the top-`K` chunks are sent — not the full document — the model
physically cannot see content outside what FAISS retrieved.

## API reference

- `POST /api/upload` — multipart form, field `file` (PDF). Returns
  `{status, filename, pages_processed, chunks_created}`.
- `POST /api/ask` — JSON `{"question": "...", "top_k": 5}` (`top_k` optional).
  Returns `{"answer": "...", "sources": [{"page", "similarity", "text", "source"}]}`.
- `GET /api/status` — `{"ready": bool, "num_chunks": int}`.
