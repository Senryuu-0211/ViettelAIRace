"""
Embedding client cho ICD linker — gọi Ollama, cache theo tên model.
Đổi OLLAMA_EMBED_MODEL trong config để thử model nhẹ hơn; cache tách riêng
mỗi model nên chuyển qua lại không phải tính lại từ đầu.
"""
import hashlib
import os

import numpy as np

from pipeline.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, EMBED_CACHE_DIR


def _l2norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def _client(model: str):
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)


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

    client = _client(model)
    vecs = np.asarray(client.embed_documents(texts), dtype=np.float32)
    vecs = _l2norm(vecs)
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, v=vecs)
    return vecs


def embed_query(text: str, model: str = None) -> np.ndarray:
    """Embed + L2-normalize một query (vector 1 chiều)."""
    model = model or OLLAMA_EMBED_MODEL
    v = np.asarray(_client(model).embed_query(text), dtype=np.float32)
    return _l2norm(v[None, :])[0]
