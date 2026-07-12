"""
Embedding client cho ICD linker — gọi THẲNG Ollama /api/embeddings qua urllib.
(Không dùng langchain_ollama: client đó gọi /tokenize và lỗi trên bản mới.)
Cache theo tên model + nội dung -> đổi OLLAMA_EMBED_MODEL để test model khác,
mỗi model cache riêng nên chuyển qua lại không phải tính lại.
"""
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np

from pipeline.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, EMBED_CACHE_DIR


def _l2norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    """Batch embed qua /api/embed (nhanh hơn nhiều so với gọi từng text)."""
    body = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["embeddings"]


def _embed_one(text: str, model: str) -> list[float]:
    return _embed_batch([text], model)[0]


def _cache_path(model: str, cache_key: str, texts_hash: str) -> str:
    safe = model.replace(":", "_").replace("/", "_")
    return os.path.join(EMBED_CACHE_DIR, f"{safe}__{cache_key}__{texts_hash}.npz")


def embed_corpus(texts: list[str], cache_key: str, model: str = None) -> np.ndarray:
    """Embed + L2-normalize danh sách text; cache ra đĩa (theo model + nội dung)."""
    model = model or OLLAMA_EMBED_MODEL
    h = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    path = _cache_path(model, cache_key, h)
    if os.path.exists(path):
        return np.load(path)["v"]

    rows = []
    batch = 128
    for i in range(0, len(texts), batch):
        rows.extend(_embed_batch(texts[i:i + batch], model))
        print(f"    embed {min(i + batch, len(texts))}/{len(texts)}", file=sys.stderr, flush=True)
    vecs = _l2norm(np.asarray(rows, dtype=np.float32))
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, v=vecs)
    return vecs


def embed_query(text: str, model: str = None) -> np.ndarray:
    """Embed + L2-normalize một query (vector 1 chiều)."""
    model = model or OLLAMA_EMBED_MODEL
    v = np.asarray(_embed_one(text, model), dtype=np.float32)
    return _l2norm(v[None, :])[0]
