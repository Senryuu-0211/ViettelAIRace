#!/usr/bin/env python3
"""
Medical Concept Extraction Pipeline — LangChain-style LLM approach.
Processes clinical notes in input/ -> output/N.json.
"""

import argparse
import json
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pipeline.config import (
    INPUT_DIR,
    OUTPUT_DIR,
    ASSERTION_VALID_TYPES,
    CANDIDATE_VALID_TYPES,
    VALID_TYPES,
    VALID_ASSERTIONS,
)
from pipeline.extractors.entity_extractor import extract_entities, extract_entities_batch
from pipeline.extractors.assertion_extractor import extract_assertions_batch
from pipeline.linker.rxnorm import lookup_drug
from pipeline.linker.icd10 import lookup_diagnosis
from pipeline.postprocess.position import find_position
from pipeline.postprocess.duplicate import expand_duplicates


def process_document(input_text: str, doc_id: str, skip_assertion: bool = False,
                     skip_linker: bool = False, skip_position: bool = False,
                     expand_dup: bool = True) -> list[dict]:
    t0 = time.time()

    print("  [1/4] Extracting entities ...", end=" ", flush=True)
    entities = extract_entities(input_text)
    orig_count = len(entities)
    print(f"{orig_count} found")

    if expand_dup and orig_count > 0:
        expanded = expand_duplicates(entities, input_text)
        added = len(expanded) - orig_count
        if added > 0:
            print(f"        Expanded duplicates: +{added} (total {len(expanded)})")
        entities = expanded

    if skip_position:
        print("  [SKIP] Position")
    else:
        print("  [2/4] Computing positions ...", end=" ", flush=True)
        used_ranges = []
        for e in entities:
            pos = find_position(input_text, e["text"], used_ranges)
            e["position"] = pos if pos else []
        print("done")

    if skip_assertion:
        print("  [SKIP] Assertions")
        for e in entities:
            e["assertions"] = []
    else:
        print("  [3/4] Classifying assertions (batch) ...", end=" ", flush=True)
        batch_results = extract_assertions_batch(input_text, entities)
        count = 0
        for e in entities:
            if e["text"] in batch_results:
                e["assertions"] = batch_results[e["text"]]
                if batch_results[e["text"]]:
                    count += 1
            else:
                e["assertions"] = []
        print(f"{count} with assertions")

    if skip_linker:
        print("  [SKIP] Candidates")
        for e in entities:
            e["candidates"] = []
    else:
        print("  [4/4] Linking candidates ...", end=" ", flush=True)
        linked = 0
        for e in entities:
            if e["type"] == "THUỐC":
                codes = lookup_drug(e["text"], e.get("drug"))
                e["candidates"] = codes
                if codes:
                    linked += 1
            elif e["type"] == "CHẨN_ĐOÁN":
                codes = lookup_diagnosis(e["text"])
                e["candidates"] = codes
                if codes:
                    linked += 1
            else:
                e["candidates"] = []
        print(f"{linked} linked")

    _validate_entities(entities)

    elapsed = time.time() - t0
    print(f"  -> {len(entities)} concepts, {elapsed:.1f}s")
    return entities


def _validate_entities(entities: list[dict]):
    for e in entities:
        e["type"] = e.get("type", "")
        e["assertions"] = e.get("assertions", [])
        e["candidates"] = e.get("candidates", [])
        e["position"] = e.get("position", [])
        assert e["type"] in VALID_TYPES, f"Invalid type: {e['type']}"
        assert isinstance(e["assertions"], list), f"Assertions must be list: {e}"
        for a in e["assertions"]:
            assert a in VALID_ASSERTIONS, f"Invalid assertion: {a}"
        assert isinstance(e["candidates"], list), f"Candidates must be list: {e}"
        if e["type"] not in CANDIDATE_VALID_TYPES:
            e.pop("candidates", None)


def process_file(filename: str, **kwargs) -> dict:
    input_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        input_text = f.read()

    doc_id = filename.replace(".txt", "")
    entities = process_document(input_text, doc_id, **kwargs)
    _strip_drug(entities)

    out_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

    return {"id": doc_id, "entities": len(entities), "ok": True, "out": out_path}


def process_range(start: int, end: int, resume: bool = False, batch_size: int = 1, **kwargs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_ids = list(range(start, end + 1))
    results = []
    t_total = time.time()

    idx = 0
    while idx < len(file_ids):
        batch_ids = file_ids[idx:idx + batch_size]
        batch_texts = []
        batch_meta = []

        for i in batch_ids:
            filename = f"{i}.txt"
            out_path = os.path.join(OUTPUT_DIR, f"{i}.json")

            if resume and os.path.exists(out_path):
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    n = len(existing) if isinstance(existing, list) else 0
                except Exception:
                    n = 0
                print(f"[{len(results)+1}/{end-start+1}] {filename} SKIP (already {n} entities)")
                results.append({"id": str(i), "entities": n, "ok": True, "out": out_path, "skipped": True})
                continue

            input_path = os.path.join(INPUT_DIR, filename)
            if not os.path.exists(input_path):
                print(f"[{len(results)+1}/{end-start+1}] {filename} not found, SKIP")
                results.append({"id": str(i), "entities": 0, "ok": False, "out": "", "skipped": True})
                continue

            with open(input_path, "r", encoding="utf-8") as f:
                batch_texts.append(f.read())
            batch_meta.append((i, filename))

        if not batch_texts:
            idx += batch_size
            continue

        t_batch = time.time()
        file_nums = ",".join(str(i) for i, _ in batch_meta)
        print(f"\n{'=' * 60}")
        print(f"  Batch: files [{file_nums}] ({len(batch_texts)} files)")
        print(f"{'=' * 60}")

        all_entities = extract_entities_batch(batch_texts)

        pbar = tqdm(zip(batch_meta, all_entities), desc="  Processing", unit="file",
                    total=len(batch_meta))
        for tidx, ((i, filename), entities) in enumerate(pbar):
            pbar.set_postfix_str(f"file={filename}")
            doc_id = str(i)
            out_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
            input_text = batch_texts[tidx]

            print(f"\n  [{len(results)+1}/{end-start+1}] Processing: {filename}")
            try:
                output_entities = _process_entities(entities, input_text, doc_id, **kwargs)
                _strip_drug(output_entities)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(output_entities, f, ensure_ascii=False, indent=2)
                results.append({"id": doc_id, "entities": len(output_entities), "ok": True,
                                "out": out_path, "skipped": False})
            except Exception as e:
                print(f"  [FAIL] {e}")
                results.append({"id": doc_id, "entities": 0, "ok": False,
                                "out": "", "skipped": False})

        t_batch = time.time() - t_batch
        print(f"\n  Batch [{file_nums}] done in {t_batch:.1f}s")
        idx += batch_size

    t_total = time.time() - t_total
    _print_summary(results, t_total)


def _process_entities(entities: list[dict], input_text: str, doc_id: str,
                      skip_assertion: bool = False, skip_linker: bool = False,
                      skip_position: bool = False, expand_dup: bool = True) -> list[dict]:
    t0 = time.time()
    orig_count = len(entities)
    print(f"  [1/4] Entities: {orig_count} found")

    if expand_dup and orig_count > 0:
        expanded = expand_duplicates(entities, input_text)
        added = len(expanded) - orig_count
        if added > 0:
            print(f"        Expanded duplicates: +{added} (total {len(expanded)})")
        entities = expanded

    if skip_position:
        print("  [SKIP] Position")
    else:
        print("  [2/4] Computing positions ...", end=" ", flush=True)
        used_ranges = []
        for e in entities:
            pos = find_position(input_text, e["text"], used_ranges)
            e["position"] = pos if pos else []
        print("done")

    if skip_assertion:
        print("  [SKIP] Assertions")
        for e in entities:
            e["assertions"] = []
    else:
        print("  [3/4] Classifying assertions (batch) ...", end=" ", flush=True)
        batch_results = extract_assertions_batch(input_text, entities)
        count = 0
        for e in entities:
            if e["text"] in batch_results:
                e["assertions"] = batch_results[e["text"]]
                if batch_results[e["text"]]:
                    count += 1
            else:
                e["assertions"] = []
        print(f"{count} with assertions")

    if skip_linker:
        print("  [SKIP] Candidates")
        for e in entities:
            e["candidates"] = []
    else:
        print("  [4/4] Linking candidates ...", end=" ", flush=True)
        linked = 0
        for e in entities:
            if e["type"] == "THUỐC":
                codes = lookup_drug(e["text"], e.get("drug"))
                e["candidates"] = codes
                if codes:
                    linked += 1
            elif e["type"] == "CHẨN_ĐOÁN":
                codes = lookup_diagnosis(e["text"])
                e["candidates"] = codes
                if codes:
                    linked += 1
            else:
                e["candidates"] = []
        print(f"{linked} linked")

    _validate_entities(entities)

    elapsed = time.time() - t0
    print(f"  -> {len(entities)} concepts, {elapsed:.1f}s")
    return entities


def _strip_drug(entities: list[dict]):
    for e in entities:
        e.pop("drug", None)


def _print_summary(results: list[dict], total_time: float):
    ok = sum(1 for r in results if r["ok"])
    fail = len(results) - ok
    total_entities = sum(r["entities"] for r in results)
    processed = [r for r in results if not r.get("skipped")]

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"  Files: {len(results)} total, {ok} ok, {fail} fail")
    if processed:
        print(f"  Processed: {len(processed)} files, {total_entities} entities")
        avg = total_time / len(processed) if processed else 0
        print(f"  Time: {total_time:.1f}s total, {avg:.1f}s avg/file")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Medical Concept Extraction Pipeline")
    parser.add_argument("--file", type=str, help="Process a single file (e.g., 1.txt)")
    parser.add_argument("--range", type=str, help="Process range (e.g., 1-5)")
    parser.add_argument("--all", action="store_true", help="Process all 100 files")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch N files for entity extraction (default 10, 1=sequential)")
    parser.add_argument("--resume", action="store_true", help="Skip files with existing output")
    parser.add_argument("--skip-assertion", action="store_true", help="Skip assertion step")
    parser.add_argument("--skip-linker", action="store_true", help="Skip candidate linking")
    parser.add_argument("--skip-position", action="store_true", help="Skip position extraction")
    parser.add_argument("--no-expand", action="store_true", help="Disable duplicate expansion")
    args = parser.parse_args()

    kwargs = {
        "skip_assertion": args.skip_assertion,
        "skip_linker": args.skip_linker,
        "skip_position": args.skip_position,
        "expand_dup": not args.no_expand,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.file:
        r = process_file(args.file, **kwargs)
        _print_summary([r], 0)
    elif args.range:
        parts = args.range.split("-")
        start, end = int(parts[0]), int(parts[-1])
        process_range(start, end, resume=args.resume, batch_size=args.batch_size, **kwargs)
    elif args.all:
        process_range(1, 100, resume=args.resume, batch_size=args.batch_size, **kwargs)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
