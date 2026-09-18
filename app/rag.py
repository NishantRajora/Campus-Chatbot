"""
Ties retrieval + the strict prompt + the Ollama Cloud API call together.
"""
import requests

from app.config import OLLAMA_API_KEY, OLLAMA_API_URL, OLLAMA_MODEL, TOP_K
from app.embeddings import embed_query
from app.vector_store import store

NOT_FOUND_MESSAGE = "I could not find this information in the provided document."

SYSTEM_PROMPT = """You are a document-based question answering assistant.

Answer ONLY using the provided context.

Do NOT use outside knowledge.

Do NOT guess or invent information.

If the answer cannot be found in the provided context,
respond exactly:

"I could not find this information in the provided document."

If the context provides only partial information,
clearly state that the information is partial.

Always prefer the retrieved document content over
your own knowledge."""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Page {c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def build_user_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion:\n{question}"


def call_ollama(system_prompt: str, user_prompt: str) -> str:
    if not OLLAMA_API_KEY or OLLAMA_API_KEY == "xxx":
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Add it to your .env file (never to frontend code)."
        )

    url = f"{OLLAMA_API_URL.rstrip('/')}/api/chat"
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    # Ollama's /api/chat returns {"message": {"role": "...", "content": "..."}}
    message = data.get("message", {})
    content = message.get("content", "").strip()
    if not content:
        raise RuntimeError(f"Empty response from Ollama API: {data}")
    return content


def answer_question(question: str, top_k: int | None = None) -> dict:
    """
    Full pipeline: embed question -> FAISS search -> strict prompt -> Ollama.
    Returns {"answer": str, "sources": [...]}.
    """
    if not store.is_ready():
        return {
            "answer": "No document has been uploaded yet. Please upload a PDF first.",
            "sources": [],
        }

    k = top_k or TOP_K
    query_vec = embed_query(question)
    retrieved = store.search(query_vec, k)

    if not retrieved:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    context = build_context(retrieved)
    user_prompt = build_user_prompt(context, question)
    answer = call_ollama(SYSTEM_PROMPT, user_prompt)

    sources = [
        {
            "page": r["page"],
            "similarity": round(r["similarity"], 4),
            "text": r["text"][:400],
            "source": r.get("source", ""),
        }
        for r in retrieved
    ]
    return {"answer": answer, "sources": sources}
