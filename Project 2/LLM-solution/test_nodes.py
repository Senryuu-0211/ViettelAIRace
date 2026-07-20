# -*- coding: utf-8 -*-
"""
Test individual LangGraph extraction nodes.

Usage:
  python test_nodes.py --file 1.txt                          # all 5 nodes
  python test_nodes.py --file 1.txt --node thuoc              # single node
  python test_nodes.py --file 1.txt --node thuoc,chandoan     # specific nodes
  python test_nodes.py --file 1.txt --node all --compare      # + graph.invoke() comparison
  python test_nodes.py --file 1.txt --num-ctx 16384           # override context
  python test_nodes.py --file 1.txt --node thuoc --raw        # print full LLM output
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.config import INPUT_DIR, OLLAMA_MODEL
from pipeline.langgraph_extractor.graph import create_extractor_chain, parse_llm_json, build_extraction_graph
from pipeline.langgraph_extractor.prompts.thuoc_prompt import THUOC_SYSTEM_PROMPT, THUOC_USER_TEMPLATE
from pipeline.langgraph_extractor.prompts.chandoan_prompt import CHANDOAN_SYSTEM_PROMPT, CHANDOAN_USER_TEMPLATE
from pipeline.langgraph_extractor.prompts.trieuchung_prompt import TRIEUCHUNG_SYSTEM_PROMPT, TRIEUCHUNG_USER_TEMPLATE
from pipeline.langgraph_extractor.prompts.ten_xn_prompt import TENXN_SYSTEM_PROMPT, TENXN_USER_TEMPLATE
from pipeline.langgraph_extractor.prompts.kq_xn_prompt import KQXN_SYSTEM_PROMPT, KQXN_USER_TEMPLATE

NODES = {
    "thuoc":      ("THUỐC",               THUOC_SYSTEM_PROMPT,      THUOC_USER_TEMPLATE),
    "chandoan":   ("CHẨN_ĐOÁN",           CHANDOAN_SYSTEM_PROMPT,   CHANDOAN_USER_TEMPLATE),
    "trieuchung": ("TRIỆU_CHỨNG",         TRIEUCHUNG_SYSTEM_PROMPT, TRIEUCHUNG_USER_TEMPLATE),
    "tenxn":      ("TÊN_XÉT_NGHIỆM",      TENXN_SYSTEM_PROMPT,      TENXN_USER_TEMPLATE),
    "kqxn":       ("KẾT_QUẢ_XÉT_NGHIỆM",  KQXN_SYSTEM_PROMPT,      KQXN_USER_TEMPLATE),
}


def test_node(name: str, input_text: str, num_ctx: int = None, show_raw: bool = False) -> dict:
    display, sys_prompt, usr_tmpl = NODES[name]
    chain = create_extractor_chain(sys_prompt, usr_tmpl, num_ctx=num_ctx)

    t0 = time.time()
    result = chain.invoke({"input_text": input_text})
    elapsed = time.time() - t0

    parsed = parse_llm_json(result)
    entity_texts = [e.get("text", "") for e in parsed]

    print(f"\n{'─' * 50}")
    print(f"  Node: {display}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Content len: {len(result)} chars")
    print(f"  Parsed entities: {len(parsed)}")
    if entity_texts:
        print(f"  Entities:")
        for e in parsed:
            a = e.get("assertions", [])
            a_str = f"[{', '.join(a)}]" if a else "[]"
            print(f"    {e.get('text', '')!r:40s}  assert={a_str}")
    if show_raw:
        print(f"  --- RAW OUTPUT ---\n{result}\n  --- END RAW ---")

    return {"name": display, "time": elapsed, "entities": len(parsed),
            "content_len": len(result), "parsed": parsed}


def main():
    parser = argparse.ArgumentParser(description="Test LangGraph extraction nodes")
    parser.add_argument("--file", required=True, help="Input file name (e.g., 1.txt)")
    parser.add_argument("--node", default="all",
                        help="Node(s) to test: thuoc,chandoan,trieuchung,tenxn,kqxn or 'all'")
    parser.add_argument("--num-ctx", type=int, default=None, help="Override context window")
    parser.add_argument("--raw", action="store_true", help="Print full LLM raw output")
    parser.add_argument("--compare", action="store_true",
                        help="Also run full graph.invoke() for comparison")
    args = parser.parse_args()

    input_path = os.path.join(INPUT_DIR, args.file)
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        input_text = f.read()

    selected = list(NODES.keys()) if args.node == "all" else \
        [n.strip() for n in args.node.split(",") if n.strip() in NODES]

    if not selected:
        print(f"Invalid node(s): {args.node}. Valid: {', '.join(NODES.keys())}, all")
        sys.exit(1)

    ctx_str = str(args.num_ctx) if args.num_ctx else "default"
    print(f"Model: {OLLAMA_MODEL}  |  num_ctx: {ctx_str}")
    print(f"Input: {os.path.basename(input_path)} ({len(input_text)} chars)")

    results = []
    for name in selected:
        r = test_node(name, input_text, num_ctx=args.num_ctx, show_raw=args.raw)
        results.append(r)

    if len(results) > 1:
        total_time = sum(r["time"] for r in results)
        total_entities = sum(r["entities"] for r in results)
        fastest = min(results, key=lambda x: x["time"])
        slowest = max(results, key=lambda x: x["time"])
        print(f"\n{'═' * 50}")
        print(f"  Totals: {total_entities} entities, {total_time:.1f}s")
        print(f"  Avg/node: {total_time/len(results):.1f}s")
        print(f"  Fastest: {fastest['name']} ({fastest['time']:.1f}s)")
        print(f"  Slowest: {slowest['name']} ({slowest['time']:.1f}s)")

    if args.compare:
        print(f"\n{'═' * 50}")
        print(f"  Running full graph.invoke() for comparison...")
        graph = build_extraction_graph()
        t0 = time.time()
        final_state = graph.invoke({"input_text": input_text, "results": []})
        graph_time = time.time() - t0
        graph_entities = len(final_state["results"])
        print(f"  Graph time: {graph_time:.1f}s")
        print(f"  Graph entities: {graph_entities}")
        print(f"  Sum of nodes: {total_time:.1f}s  |  Delta: {graph_time - total_time:+.1f}s")


if __name__ == "__main__":
    main()
