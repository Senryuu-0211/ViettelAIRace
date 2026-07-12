"""
ICD-10 linker — hybrid retrieval trên KB TT06 local (data/icd10_tt06.json).

Index 2 tầng:
  - category  (vd K21 "Bệnh trào ngược dạ dày- thực quản")  -> trả CẢ cụm mã {K21.0, K21.9}
  - leaf      (vd K21.0 "...kèm viêm thực quản")            -> trả 1 mã [K21.0]

Query chung khớp category -> SET; query cụ thể khớp leaf -> mã đơn.
Điểm = fuse(lexical rapidfuzz, cosine embedding). USE_EMBEDDING=0 -> lexical thuần.
"""
import json
import re
import unicodedata

import numpy as np

from pipeline.config import (
    ICD_KB_PATH, ICD_MIN_SCORE, ICD_LEXICAL_WEIGHT, ICD_TOP_K,
    USE_EMBEDDING, EMBED_LEAVES,
)

try:
    from rapidfuzz import fuzz, process
    _HAS_RF = True
except ImportError:
    _HAS_RF = False

# viết tắt/đồng nghĩa lâm sàng hay gặp -> dạng chuẩn để lexical bắt tốt hơn
ABBREV = {
    "tha": "tăng huyết áp",
    "đtđ": "đái tháo đường",
    "copd": "bệnh phổi tắc nghẽn mạn tính",
    "gerd": "trào ngược dạ dày thực quản",
    "nmct": "nhồi máu cơ tim",
    "tbmmn": "tai biến mạch máu não",
    "suy tim ứ huyết": "suy tim sung huyết",
}

_INDEX = None  # dict: names, norms, code_lists, emb


def _strip_dagger(code: str) -> str:
    return code.replace("†", "").replace("*", "").strip()


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower().strip()
    s = s.replace("†", "").replace("*", "")
    s = re.sub(r"[\-–—/]", " ", s)          # gộp gạch nối/gạch chéo
    s = re.sub(r"[^\w\sàáảãạ]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    for k, v in ABBREV.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    return s


def _build_index():
    global _INDEX
    with open(ICD_KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    names, code_lists = [], []
    # tầng category: tên nhóm -> cả cụm mã (đã strip dagger, dedupe, giữ thứ tự)
    for cat_code, obj in data["categories"].items():
        name = (obj.get("name") or "").strip() or cat_code
        codes = list(dict.fromkeys(_strip_dagger(c) for c in obj.get("codes", []) if _strip_dagger(c)))
        if not codes:
            continue
        names.append(name)
        code_lists.append(codes)
    # tầng leaf: tên mã lá -> chính mã đó
    if EMBED_LEAVES:
        for lf in data["codes"]:
            code = _strip_dagger(lf.get("code", ""))
            name = (lf.get("name") or "").strip()
            if not code or not name:
                continue
            names.append(name)
            code_lists.append([code])

    _INDEX = {
        "names": names,
        "norms": [_normalize(n) for n in names],
        "code_lists": code_lists,
        "emb": None,
    }
    return _INDEX


def _ensure_emb():
    if _INDEX["emb"] is None and USE_EMBEDDING:
        from pipeline.linker.embedder import embed_corpus
        key = "icd_cat_leaf" if EMBED_LEAVES else "icd_cat"
        _INDEX["emb"] = embed_corpus(_INDEX["names"], key)


def _lexical_topk(qnorm: str, k: int):
    norms = _INDEX["norms"]
    if _HAS_RF:
        hits = process.extract(qnorm, norms, scorer=fuzz.token_set_ratio,
                               limit=k, processor=None)
        return [(idx, score / 100.0) for (_, score, idx) in hits]
    # fallback: token-coverage của query
    qs = set(qnorm.split())
    scored = []
    for i, n in enumerate(norms):
        ns = set(n.split())
        scored.append((i, (len(qs & ns) / len(qs)) if qs else 0.0))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def _lexical_one(qnorm: str, idx: int) -> float:
    if _HAS_RF:
        return fuzz.token_set_ratio(qnorm, _INDEX["norms"][idx]) / 100.0
    qs, ns = set(qnorm.split()), set(_INDEX["norms"][idx].split())
    return (len(qs & ns) / len(qs)) if qs else 0.0


def lookup_diagnosis(diagnosis_text: str) -> list[str]:
    if _INDEX is None:
        _build_index()

    qnorm = _normalize(diagnosis_text)
    if not qnorm:
        return []

    lex_top = _lexical_topk(qnorm, ICD_TOP_K)          # [(idx, lex_score)]
    lex_by_idx = {idx: sc for idx, sc in lex_top}

    emb_scores = None
    if USE_EMBEDDING:
        try:
            _ensure_emb()
            from pipeline.linker.embedder import embed_query
            qv = embed_query(qnorm)
            emb_scores = _INDEX["emb"] @ qv               # cosine (đã L2-norm)
            for idx in np.argsort(-emb_scores)[:ICD_TOP_K]:
                lex_by_idx.setdefault(int(idx), _lexical_one(qnorm, int(idx)))
        except Exception:
            emb_scores = None

    pool = list(lex_by_idx.keys())
    if not pool:
        return []

    lex_raw = {i: lex_by_idx[i] for i in pool}
    emb_raw = {i: (float(emb_scores[i]) if emb_scores is not None else 0.0) for i in pool}

    if emb_scores is not None:
        # min-max normalize từng tín hiệu TRONG pool -> 2 thang so sánh được,
        # nếu không embedding (cosine ~0.6) luôn thua lexical (ratio ~0.9).
        def _mm(d):
            vals = list(d.values())
            lo, hi = min(vals), max(vals)
            rng = hi - lo
            return {k: ((v - lo) / rng if rng > 1e-9 else 1.0) for k, v in d.items()}
        lexN, embN = _mm(lex_raw), _mm(emb_raw)
        w = ICD_LEXICAL_WEIGHT
        fused = {i: w * lexN[i] + (1.0 - w) * embN[i] for i in pool}
    else:
        fused = lex_raw

    best_idx = max(pool, key=lambda i: fused[i])
    # ngưỡng emit dùng điểm THÔ tốt nhất (tránh emit khi cả 2 tín hiệu đều yếu)
    if max(lex_raw[best_idx], emb_raw[best_idx]) < ICD_MIN_SCORE:
        return []
    return list(_INDEX["code_lists"][best_idx])


def lookup_batch(diagnosis_texts: list[str]) -> dict[str, list[str]]:
    result = {}
    for text in diagnosis_texts:
        codes = lookup_diagnosis(text)
        if codes:
            result[text] = codes
    return result
