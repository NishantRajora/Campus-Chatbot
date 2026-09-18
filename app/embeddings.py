"""
Wraps sentence-transformers/all-MiniLM-L6-v2.
Loaded once as a module-level singleton so we don't reload the model
on every request.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of strings. Returns float32 array, L2-normalized
    so that FAISS inner-product search behaves like cosine similarity."""
    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])
