#!/usr/bin/env python3
"""
Smoke test / benchmark cho ICD linker.
Đo hit-rate = tỉ lệ prediction CHỨA mã kỳ vọng chính, trên bộ chẩn đoán Việt hay gặp.

So sánh model embedding:
  set OLLAMA_EMBED_MODEL=bge-m3 && python tools/test_icd_linker.py
  set OLLAMA_EMBED_MODEL=paraphrase-multilingual && python tools/test_icd_linker.py
  set USE_EMBEDDING=0 && python tools/test_icd_linker.py      # lexical thuần (mốc)
"""
import os
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import OLLAMA_EMBED_MODEL, USE_EMBEDDING  # noqa: E402
from pipeline.linker.icd10 import lookup_diagnosis            # noqa: E402

# (chẩn đoán, mã kỳ vọng chính) — hit khi mã này nằm trong prediction
CASES = [
    ("tăng huyết áp", "I10"),
    ("tăng huyết áp vô căn", "I10"),
    ("bệnh trào ngược dạ dày - thực quản", "K21.9"),
    ("trào ngược dạ dày thực quản", "K21.9"),
    ("hen suyễn", "J45.9"),
    ("hen phế quản", "J45.9"),
    ("suy tim", "I50.9"),
    ("suy tim sung huyết", "I50.0"),
    ("xơ gan do rượu", "K70.3"),
    ("đái tháo đường típ 2", "E11.9"),
    ("đái tháo đường", "E11.9"),
    ("viêm phổi", "J18.9"),
    ("nhồi máu cơ tim", "I21.9"),
    ("đột quỵ", "I64"),                 # test đồng nghĩa: tai biến mạch máu não
    ("suy thận mạn", "N18.9"),
    ("thiếu máu", "D64.9"),
    ("trầm cảm", "F32.9"),
    ("hội chứng não gan", "K72.9"),
]


def main():
    mode = f"embed={OLLAMA_EMBED_MODEL}" if USE_EMBEDDING else "LEXICAL-ONLY"
    print(f"=== ICD linker benchmark ({mode}) ===")
    hits, t0 = 0, time.time()
    for q, expected in CASES:
        pred = lookup_diagnosis(q)
        ok = expected in pred
        hits += ok
        mark = "OK " if ok else "MISS"
        print(f"  [{mark}] {q:<38} -> {pred}  (kỳ vọng {expected})")
    dt = time.time() - t0
    print(f"\nHit-rate: {hits}/{len(CASES)} = {hits/len(CASES):.1%}   ({dt:.1f}s, {dt/len(CASES):.2f}s/query)")


if __name__ == "__main__":
    main()
