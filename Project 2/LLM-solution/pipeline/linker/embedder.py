"""
Embedding client cho ICD linker — gọi Ollama qua HTTP API, cache theo tên model.
Dùng httpx gọi trực tiếp http://localhost:11434/api/embed để tránh bug port của
thư viện ollama Python >=0.5.
"""
import hashlib
import os

import httpx
import numpy as np

from pipeline.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, EMBED_CACHE_DIR


def _l2norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def _call_embed(input_data, model: str):
    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": model, "input": input_data},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _cache_path(model: str, cache_key: str, texts_hash: str) -> str:
    safe = model.replace(":", "_").replace("/", "_")
    return os.path.join(EMBED_CACHE_DIR, f"{safe}__{cache_key}__{texts_hash}.npz")


def embed_corpus(texts: list[str], cache_key: str, model: str = None) -> np.ndarray:
    """Embed + L2-normalize một danh sách text; cache ra đĩa (theo model + nội dung)."""
    model = model or OLLAMA_EMBED_MODEL
    h = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    path = _cache_path(model, cache_key, h)
    if os.path.exists(path):
        return np.load(path)["v"]

    vecs = np.asarray(_call_embed(texts, model), dtype=np.float32)
    vecs = _l2norm(vecs)
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, v=vecs)
    return vecs


def embed_query(text: str, model: str = None) -> np.ndarray:
    """Embed + L2-normalize một query (vector 1 chiều)."""
    model = model or OLLAMA_EMBED_MODEL
    v = np.asarray(_call_embed(text, model)[0], dtype=np.float32)
    return _l2norm(v[None, :])[0]
