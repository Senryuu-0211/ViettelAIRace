#!/usr/bin/env python3
"""
Eval 1 sample gold thật = ví dụ drug-list trong đề (input + gold do BTC cung cấp).
Chạy pipeline (qwen3.5:4b) -> chấm bằng scorer -> in diff pred vs gold để sửa prompt.

Chạy:  <firstconda python> tools/eval_druglist.py
"""
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.main import process_document          # noqa: E402
from scorer import evaluate_sample                   # noqa: E402

INPUT = ("Danh sách thuốc trước nhập viện chính xác và đầy đủ. "
         "1. amlodipine 10 mg po daily 2. aspirin 81 mg po daily "
         "3. metoprolol succinate xl 50 mg po daily 4. guaifenesin ml po q6h:prn điều trị ho "
         "5. nystatin oral suspension 5 ml po qid:prn điều trị đau nhức "
         "6. acetaminophen 325-650 mg po q6h:prn điều trị sốt đau 7. pravastatin 40 mg po daily "
         "8. docusate sodium 100 mg po bid điều trị táo bón 9. senna 8.6 mg po bid:prn điều trị táo bón "
         "10. clonazepam 0.5 mg po qam:prn điều trị lo âu "
         "11. clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ")

GOLD = [
    {"text": "amlodipine 10 mg po daily", "type": "THUỐC", "candidates": ["308135"], "assertions": ["isHistorical"]},
    {"text": "aspirin 81 mg po daily", "type": "THUỐC", "candidates": ["243670"], "assertions": ["isHistorical"]},
    {"text": "metoprolol succinate xl 50 mg po daily", "type": "THUỐC", "candidates": ["866436"], "assertions": ["isHistorical"]},
    {"text": "guaifenesin ml po q6h:prn", "type": "THUỐC", "candidates": ["392085"], "assertions": ["isHistorical"]},
    {"text": "ho", "type": "TRIỆU_CHỨNG", "candidates": [], "assertions": []},
    {"text": "nystatin oral suspension 5 ml po qid:prn", "type": "THUỐC", "candidates": ["7597"], "assertions": ["isHistorical"]},
    {"text": "đau nhức", "type": "TRIỆU_CHỨNG", "candidates": [], "assertions": []},
    {"text": "acetaminophen 325-650 mg po q6h:prn", "type": "THUỐC", "candidates": ["313782"], "assertions": ["isHistorical"]},
    {"text": "sốt đau", "type": "TRIỆU_CHỨNG", "candidates": [], "assertions": []},
    {"text": "pravastatin 40 mg po daily", "type": "THUỐC", "candidates": ["904475"], "assertions": ["isHistorical"]},
    {"text": "docusate sodium 100 mg po bid", "type": "THUỐC", "candidates": ["1099279"], "assertions": ["isHistorical"]},
    {"text": "táo bón", "type": "TRIỆU_CHỨNG", "candidates": [], "assertions": []},
    {"text": "senna 8.6 mg po bid:prn", "type": "THUỐC", "candidates": ["312935"], "assertions": ["isHistorical"]},
    {"text": "táo bón", "type": "TRIỆU_CHỨNG", "candidates": [], "assertions": []},
    {"text": "clonazepam 0.5 mg po qam:prn", "type": "THUỐC", "candidates": ["197527"], "assertions": ["isHistorical"]},
    {"text": "lo âu", "type": "TRIỆU_CHỨNG", "candidates": [], "assertions": []},
    {"text": "clonazepam 1.5 mg po qhs", "type": "THUỐC", "candidates": ["197528"], "assertions": ["isHistorical"]},
    {"text": "lo âu", "type": "TRIỆU_CHỨNG", "candidates": [], "assertions": []},
    {"text": "mất ngủ", "type": "TRIỆU_CHỨNG", "candidates": [], "assertions": []},
]


def main():
    print(f"Input len = {len(INPUT)} chars\n")
    pred = process_document(INPUT, "druglist")

    print("\n=== PRED (pipeline) ===")
    for e in pred:
        print(f"  {e['type']:<16} {e['text']!r:<45} cand={e.get('candidates', [])} assert={e.get('assertions', [])}")

    res = evaluate_sample(pred, GOLD)
    print("\n=== SCORE (sample này) ===")
    print(f"  text:       {res['text_score']:.3f}  (x0.3)")
    print(f"  assertions: {res['assertions_score']:.3f}  (x0.3)")
    print(f"  candidates: {res['candidates_score']:.3f}  (x0.4)")
    print(f"  FINAL:      {res['final_score']:.3f}")
    print(f"  gold={res['n_gold']} pred={res['n_pred']} matched={res['n_matched']} "
          f"missed={res['n_missed']} extra={res['n_extra']}")


if __name__ == "__main__":
    main()
