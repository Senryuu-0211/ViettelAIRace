import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pipeline.config import INPUT_DIR, OUTPUT_DIR
from pipeline.langgraph_extractor.graph import build_extraction_graph
from pipeline.langgraph_extractor.postprocess import process_and_map_exact_position


def process_single(input_text: str, doc_id: str, out_dir: str) -> dict:
    t0 = time.time()

    graph = build_extraction_graph()
    initial_state = {"input_text": input_text, "results": []}
    final_state = graph.invoke(initial_state)
    raw_entities = final_state["results"]

    final_entities = process_and_map_exact_position(input_text, raw_entities)

    out_path = os.path.join(out_dir, f"{doc_id}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_entities, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    return {
        "id": doc_id,
        "entities": len(final_entities),
        "raw": len(raw_entities),
        "ok": True,
        "out": out_path,
        "time": elapsed,
    }


def process_file(filename: str, out_dir: Optional[str] = None):
    input_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        input_text = f.read()

    doc_id = filename.replace(".txt", "")
    out_dir = out_dir or OUTPUT_DIR

    print(f"Processing {filename} via LangGraph (5 models in parallel)...")
    result = process_single(input_text, doc_id, out_dir)
    print(f"  -> {result['entities']} entities (raw: {result['raw']}), {result['time']:.1f}s")
    print(f"  -> Saved to {result['out']}")


def process_range(start: int, end: int, max_workers: int = 1,
                  resume: bool = False, out_dir: Optional[str] = None):
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    file_ids = list(range(start, end + 1))
    tasks = []
    skipped = 0

    for i in file_ids:
        filename = f"{i}.txt"
        input_path = os.path.join(INPUT_DIR, filename)
        out_path = os.path.join(out_dir, f"{i}.json")

        if resume and os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                n = len(existing) if isinstance(existing, list) else 0
            except Exception:
                n = 0
            print(f"[{i}] {filename} SKIP (already {n} entities)")
            skipped += 1
            continue

        if not os.path.exists(input_path):
            print(f"[{i}] {filename} not found, SKIP")
            skipped += 1
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            input_text = f.read()
        tasks.append((str(i), input_text))

    if not tasks:
        print(f"All {skipped} files already processed or missing. Nothing to do.")
        return

    total = len(file_ids)
    n_tasks = len(tasks)
    print(f"\n{'=' * 60}")
    print(f"  Batch mode: {n_tasks} files to process")
    print(f"  Workers: {max_workers} parallel")
    if resume:
        print(f"  Resumed: {skipped} files skipped")
    print(f"{'=' * 60}\n")

    results = []
    t_total = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, text, doc_id, out_dir): doc_id
            for doc_id, text in tasks
        }

        p_done = skipped
        for future in as_completed(futures):
            p_done += 1
            try:
                r = future.result()
                results.append(r)
                print(f"  [{p_done}/{total}] {r['id']}.txt -> "
                      f"{r['entities']} entities ({r['time']:.1f}s)")
            except Exception as e:
                doc_id = futures[future]
                print(f"  [{p_done}/{total}] {doc_id}.txt -> FAIL: {e}")
                results.append({"id": doc_id, "entities": 0, "ok": False, "out": "", "time": 0})

    t_total = time.time() - t_total
    _print_summary(results, total, skipped, t_total)


def _print_summary(results: list[dict], total: int, skipped: int, total_time: float):
    ok = sum(1 for r in results if r["ok"])
    fail = sum(1 for r in results if not r["ok"])
    total_entities = sum(r["entities"] for r in results)

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"  Files: {total} total, {skipped} skipped, {ok} ok, {fail} fail")
    if ok:
        avg_time = sum(r.get("time", 0) for r in results) / max(ok, 1)
        print(f"  Entities: {total_entities} total ({total_entities // max(ok, 1)} avg/file)")
        print(f"  Time: {total_time:.1f}s total, {avg_time:.1f}s avg/file")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Medical Concept Extraction - LangGraph")
    parser.add_argument("--file", type=str, help="Process a single file (e.g., 1.txt)")
    parser.add_argument("--range", type=str, help="Process range (e.g., 1-10)")
    parser.add_argument("--all", action="store_true", help="Process all .txt files in input/")
    parser.add_argument("--workers", type=int, default=2,
                        help="Max parallel files (default 2). More workers = faster but more VRAM")
    parser.add_argument("--resume", action="store_true", help="Skip files with existing output")
    parser.add_argument("--out", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    if args.file:
        process_file(args.file, out_dir=args.out)
    elif args.range:
        parts = args.range.split("-")
        start, end = int(parts[0]), int(parts[-1])
        process_range(start, end, max_workers=args.workers,
                      resume=args.resume, out_dir=args.out)
    elif args.all:
        input_files = sorted([
            f for f in os.listdir(INPUT_DIR)
            if f.endswith(".txt")
        ])
        if not input_files:
            print(f"No .txt files found in {INPUT_DIR}")
            return
        ids = sorted(int(f.replace(".txt", "")) for f in input_files)
        process_range(ids[0], ids[-1], max_workers=args.workers,
                      resume=args.resume, out_dir=args.out)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
