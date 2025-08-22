import os
import requests
from pathlib import Path
import numpy as np

from typing import cast, Sequence
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI

from pydantic import BaseModel
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings


class DummyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Devolvemos un vector “dummy” por cada documento de entrada.
        # No se usa realmente porque pasamos embeddings manuales en .add(...)
        return [[0.0] for _ in input]


import chromadb

# Chroma
from chromadb.config import Settings

# Embeddings BGE-M3
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "../chroma_storage")
CHROMA_PATH = str((Path(__file__).resolve().parent / CHROMA_DIR))
COLLECTION = "company_docs"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings())
collection = client.get_or_create_collection(
    name=COLLECTION,
    embedding_function=DummyEmbeddingFunction(),
)

embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


app = FastAPI(title="Company Assistant (RAG + Chroma)")


class AskRequest(BaseModel):
    query: str
    k: int = 5


class Source(BaseModel):
    source: str
    chunk: int


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


def embed_query(q: str) -> List[float]:
    # La salida de BGEM3 puede ser list o ndarray; normalizamos a list[float]
    res = embed_model.encode([q], batch_size=1)
    dense = res["dense_vecs"]  # type: ignore[index]
    # dense debería ser una lista de vectores; tomamos el primero y lo convertimos a float32 list
    vec = np.asarray(dense[0], dtype=np.float32).tolist()
    return cast(List[float], vec)


def retrieve(query: str, k: int = 5) -> List[Dict]:
    qvec: List[float] = embed_query(query)

    # Chroma acepta list[list[float]] o ndarray; le pasamos list[list[float]]
    res = collection.query(
        query_embeddings=[qvec],  # type: ignore[arg-type]  # (Pyright a veces se pone exquisito)
        n_results=k,
        include=["documents", "metadatas", "embeddings"],
    )

    # Manejo seguro de None y formas vacías
    docs_ll = res.get("documents") or [[]]
    metas_ll = res.get("metadatas") or [[]]

    docs: Sequence[str] = cast(Sequence[str], docs_ll[0] if len(docs_ll) > 0 else [])
    metas: Sequence[Dict] = cast(
        Sequence[Dict], metas_ll[0] if len(metas_ll) > 0 else []
    )

    out: List[Dict] = []
    for d, m in zip(docs, metas):
        out.append(
            {
                "text": d,
                "source": m.get("source", ""),
                "chunk_idx": m.get("chunk_idx", 0),
            }
        )
    return out


def build_prompt(query: str, contexts: List[Dict]) -> str:
    ctx_lines = []
    for c in contexts:
        tag = f"[{c['source']}#{c['chunk_idx']}]"
        ctx_lines.append(f"{tag}\n{c['text']}")
    system = (
        "You are an internal assistant. "
        "Answer ONLY using the CONTEXT. "
        "If the answer is not present, say you don't know. "
        "Cite sources as [path#chunk]."
    )
    user = f"QUESTION:\n{query}\n\nCONTEXT:\n" + "\n\n".join(ctx_lines)
    return f"{system}\n\n{user}"


def call_llm(prompt: str) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    ctx = retrieve(req.query, req.k)
    prompt = build_prompt(req.query, ctx)
    answer = call_llm(prompt)
    sources_list = [Source(source=c["source"], chunk=c["chunk_idx"]) for c in ctx]
    return AskResponse(answer=answer, sources=sources_list)
