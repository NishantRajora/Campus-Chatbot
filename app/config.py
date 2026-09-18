"""
Centralized configuration loaded from .env
Nothing here is hardcoded - the API key never touches the frontend.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "https://ollama.com")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

TOP_K = int(os.getenv("TOP_K", 5))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"
UPLOADS_DIR = BASE_DIR / "uploads"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.pkl"

VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

if not OLLAMA_API_KEY or OLLAMA_API_KEY == "xxx":
    # Not raising here - lets the server boot so /health works even before
    # the .env is filled in. The chat endpoint checks this again and fails loudly.
    pass
