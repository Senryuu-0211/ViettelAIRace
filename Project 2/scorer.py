#!/usr/bin/env python3
"""
Offline scorer for Medical Concept Extraction Pipeline.
Evaluates pred vs gold per sample, matching entities by exact (text, type).
"""

import argparse
import json
import os
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def wer(ref: str, hyp: str) -> float:
    """Word Error Rate between two strings. 0 = perfect match."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    n = len(ref_words)
    if n == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dp = list(range(n + 1))
    for i, hw in enumerate(hyp_words, 1):
        prev = dp[0]
        dp[0] = i
        for j, rw in enumerate(ref_words, 1):
            old = dp[j]
            if rw == hw:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = old
    return dp[n] / n


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity with empty-set rule."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


ASSERTION_LABELS = {"isNegated", "isHistorical", "isFamily"}
ASSERTION_TYPES = {"THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}
CANDIDATE_TYPES = {"THUỐC", "CHẨN_ĐOÁN"}

WEIGHT_TEXT = 0.3
WEIGHT_ASSERTIONS = 0.3
WEIGHT_CANDIDATES = 0.4


# ---------------------------------------------------------------------------
# Align
# ---------------------------------------------------------------------------

def assertions_set(entity: dict) -> set:
    """Convert assertion to set of active label names. Supports both formats."""
    assertions = entity.get("assertions", entity.get("assertion", []))

    if isinstance(assertions, list):
        return {a for a in assertions if a in ASSERTION_LABELS}

    if isinstance(assertions, dict):
        return {k for k in ASSERTION_LABELS if assertions.get(k) is True}

    return set()


def candidates_set(entity: dict) -> set:
    """Return candidate codes as a set (lowercased, stripped)."""
    return {c.strip().lower() for c in entity.get("candidates", []) if c}


def align_entities(pred_entities: list, gold_entities: list):
    """
    Match pred entities to gold entities by exact (text, type).
    Returns:
      aligned: list of (pred_idx, gold_idx) pairs
      unmatched_pred: set of pred indices not matched
      unmatched_gold: set of gold indices not matched
    """
    used_pred = set()
    aligned = []

    for g_idx, gold in enumerate(gold_entities):
        g_text = gold["text"]
        g_type = gold["type"]
        matched = None
        for p_idx, pred in enumerate(pred_entities):
            if p_idx in used_pred:
                continue
            if pred["type"] != g_type:
                continue
            if pred["text"] == g_text:
                # Also check: same-text-different-type means BOTH penalized.
                # This is handled because wrong-type pred won't match here
                # (type already checked above), so gold gets 0 and pred stays
                # unmatched (also 0).
                matched = p_idx
                break

        if matched is not None:
            used_pred.add(matched)
            aligned.append((matched, g_idx))

    unmatched_pred = set(range(len(pred_entities))) - used_pred
    unmatched_gold = set(range(len(gold_entities))) - {g for _, g in aligned}

    return aligned, unmatched_pred, unmatched_gold


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------

def evaluate_sample(pred_entities: list, gold_entities: list) -> dict:
    """Score a single sample. Returns component scores and final."""
    n_gold = len(gold_entities)
    n_pred = len(pred_entities)

    aligned, unmatched_pred, unmatched_gold = align_entities(pred_entities, gold_entities)

    # Text score (WER per entity)
    text_sum = 0.0
    for p_idx, g_idx in aligned:
        w = wer(gold_entities[g_idx]["text"], pred_entities[p_idx]["text"])
        text_sum += (1.0 - w)

    # Assertions score (Jaccard per entity)
    assert_sum = 0.0
    for p_idx, g_idx in aligned:
        gold_type = gold_entities[g_idx]["type"]
        if gold_type in ASSERTION_TYPES:
            a_pred = assertions_set(pred_entities[p_idx])
            a_gold = assertions_set(gold_entities[g_idx])
            assert_sum += jaccard(a_pred, a_gold)
        else:
            assert_sum += 1.0  # both empty

    # Candidates score (Jaccard per entity)
    cand_sum = 0.0
    for p_idx, g_idx in aligned:
        gold_type = gold_entities[g_idx]["type"]
        if gold_type in CANDIDATE_TYPES:
            c_pred = candidates_set(pred_entities[p_idx])
            c_gold = candidates_set(gold_entities[g_idx])
            cand_sum += jaccard(c_pred, c_gold)
        else:
            cand_sum += 1.0  # both empty

    # Unmatched gold entities = 0 for all metrics
    # Unmatched pred entities = 0 for all metrics
    # Denominator = number of gold entities (implicitly, unmatched = 0)

    n = n_gold if n_gold > 0 else 1

    text_score = text_sum / n
    assert_score = assert_sum / n
    cand_score = cand_sum / n
    final = WEIGHT_TEXT * text_score + WEIGHT_ASSERTIONS * assert_score + WEIGHT_CANDIDATES * cand_score

    return {
        "text_score": text_score,
        "assertions_score": assert_score,
        "candidates_score": cand_score,
        "final_score": final,
        "n_gold": n_gold,
        "n_pred": n_pred,
        "n_matched": len(aligned),
        "n_missed": len(unmatched_gold),
        "n_extra": len(unmatched_pred),
    }


# ---------------------------------------------------------------------------
# Evaluate directory of samples
# ---------------------------------------------------------------------------

def load_entities(path: str) -> list:
    """Load entities from JSON file. Supports list or {"concepts": [...]}."""
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("concepts", data.get("entities", []))
    return []


def evaluate_directory(pred_dir: str, gold_dir: str) -> dict:
    """Evaluate all samples found in gold_dir."""
    results = []
    gold_files = sorted([
        f for f in os.listdir(gold_dir)
        if f.endswith(".json")
    ])

    for gfile in gold_files:
        pfile = os.path.join(pred_dir, gfile)
        gpath = os.path.join(gold_dir, gfile)

        if not os.path.exists(pfile):
            print(f"  [WARN] Missing pred: {gfile}, treated as empty", file=sys.stderr)
            pred = []
        else:
            pred = load_entities(pfile)

        gold = load_entities(gpath)
        res = evaluate_sample(pred, gold)
        res["sample"] = gfile
        results.append(res)

    if not results:
        raise ValueError("No gold JSON files found.")

    n_samples = len(results)
    avg = {
        "text_score": sum(r["text_score"] for r in results) / n_samples,
        "assertions_score": sum(r["assertions_score"] for r in results) / n_samples,
        "candidates_score": sum(r["candidates_score"] for r in results) / n_samples,
    }
    avg["final_score"] = (
        WEIGHT_TEXT * avg["text_score"]
        + WEIGHT_ASSERTIONS * avg["assertions_score"]
        + WEIGHT_CANDIDATES * avg["candidates_score"]
    )

    return {
        "n_samples": n_samples,
        "summary": avg,
        "per_sample": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Medical Concept Extraction Scorer")
    parser.add_argument("pred", help="Prediction JSON file OR directory")
    parser.add_argument("gold", help="Gold JSON file OR directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-sample details")
    args = parser.parse_args()

    if os.path.isdir(args.pred) and os.path.isdir(args.gold):
        result = evaluate_directory(args.pred, args.gold)
    elif os.path.isfile(args.pred) and os.path.isfile(args.gold):
        pred = load_entities(args.pred)
        gold = load_entities(args.gold)
        per_sample = evaluate_sample(pred, gold)
        per_sample["sample"] = os.path.basename(args.gold)
        result = {
            "n_samples": 1,
            "summary": {
                "text_score": per_sample["text_score"],
                "assertions_score": per_sample["assertions_score"],
                "candidates_score": per_sample["candidates_score"],
                "final_score": per_sample["final_score"],
            },
            "per_sample": [per_sample],
        }
    else:
        print("Error: pred and gold must both be files or both be directories.", file=sys.stderr)
        sys.exit(1)

    s = result["summary"]
    print("=" * 60)
    print(f"  Samples evaluated:       {result['n_samples']}")
    print(f"  Text score (x0.3):       {s['text_score']:.6f}  = {WEIGHT_TEXT * s['text_score']:.6f}")
    print(f"  Assertions score (x0.3): {s['assertions_score']:.6f}  = {WEIGHT_ASSERTIONS * s['assertions_score']:.6f}")
    print(f"  Candidates score (x0.4): {s['candidates_score']:.6f}  = {WEIGHT_CANDIDATES * s['candidates_score']:.6f}")
    print("-" * 60)
    print(f"  FINAL SCORE:             {s['final_score']:.6f}")
    print("=" * 60)

    if args.verbose:
        print()
        for r in result["per_sample"]:
            print(f"  {r['sample']}:")
            print(f"    gold={r['n_gold']}  pred={r['n_pred']}  "
                  f"matched={r['n_matched']}  missed={r['n_missed']}  extra={r['n_extra']}  "
                  f"final={r['final_score']:.6f}")


if __name__ == "__main__":
    main()
