"""
Local vector database on top of FAISS.

index.faiss  -> the searchable vectors (FAISS only knows numbers)
chunks.pkl   -> the actual text + metadata, keyed by the same row order
                as the FAISS index, e.g.
                {"text": "...", "page": 25, "source": "uploaded_document.pdf"}

FAISS stores vectors, not text - chunks.pkl is what lets us turn a
vector match back into a human-readable chunk with a page number.
"""
import pickle
import faiss
import numpy as np

from app.config import FAISS_INDEX_PATH, CHUNKS_PATH


class VectorStore:
    def __init__(self):
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []
        self._load_if_exists()

    def _load_if_exists(self):
        if FAISS_INDEX_PATH.exists() and CHUNKS_PATH.exists():
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(CHUNKS_PATH, "rb") as f:
                self.chunks = pickle.load(f)

    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def build(self, embeddings: np.ndarray, chunks: list[dict]):
        """(Re)build the index from scratch for a freshly uploaded document."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product == cosine, since vectors are normalized
        index.add(embeddings)
        self.index = index
        self.chunks = chunks
        self._save()

    def _save(self):
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        if not self.is_ready():
            return []
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["similarity"] = float(score)
            results.append(chunk)
        return results


# Module-level singleton so the index loaded from disk is reused across requests
store = VectorStore()
