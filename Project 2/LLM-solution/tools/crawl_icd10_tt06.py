#!/usr/bin/env python3
"""
Crawl ICD-10 TT06 (Vietnamese, Bộ Y tế) từ backend công khai của icd.kcb.vn
(ccs.whiteneuron.com) thành 1 KB JSON local sạch, reproducible.

Cây: chapter -> section (nhóm A00-A09) -> type (category 3 ký tự) -> disease (mã lá).
API:
  GET /root                         -> danh sách chapter
  GET /childs/<parent_model>?id=<parent_id> -> danh sách con

Output: data/icd10_tt06.json
  {
    "meta": {...},
    "codes":   [{"code","name","chapter","section","category"}, ...],   # mã lá
    "categories": { "<cat_code>": {"name","codes":[...]} }              # gom theo nhóm 3 ký tự
  }
Chạy 1 lần rồi commit output → inference offline, không phụ thuộc API live.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://ccs.whiteneuron.com/api/ICD10_TT06"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "icd10_tt06.json")
WORKERS = 6
LEVEL_KEY = {"chapter": "chapter", "section": "section", "type": "category"}


def api_get(path: str):
    url = f"{BASE}/{path}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json", "User-Agent": "icd-kb-crawler"}
            )
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return None


def get_root():
    d = api_get("root")
    return d.get("data", []) if d and d.get("status") == "success" else []


def get_childs(model: str, node_id: str):
    eid = urllib.parse.quote(node_id)
    d = api_get(f"childs/{model}?id={eid}")
    return d.get("data", []) if d and d.get("status") == "success" else []


def main():
    t0 = time.time()
    leaves = []            # list of dicts
    category_names = {}    # cat_code -> tên node "type" (category)
    internal = get_root()  # frontier of non-leaf nodes to expand
    # attach ancestry to each frontier node
    frontier = [(n, {}) for n in internal]
    level = 0

    while frontier:
        level += 1
        # split leaves vs to-expand at this level, and compute ancestry for children
        to_expand = []
        for node, anc in frontier:
            model = node["model"]
            code = node["data"]["code"]
            name = (node["data"].get("name") or "").strip()
            if model == "type":
                category_names[code] = name
            anc2 = dict(anc)
            key = LEVEL_KEY.get(model)
            if key:
                anc2[key] = code
            if node.get("is_leaf"):
                leaves.append({
                    "code": code,
                    "name": (node["data"].get("name") or "").strip(),
                    "chapter": anc2.get("chapter", ""),
                    "section": anc2.get("section", ""),
                    "category": anc2.get("category", code),
                })
            else:
                to_expand.append((node, anc2))

        if not to_expand:
            break

        # fetch children of all non-leaf nodes at this level in parallel
        def fetch(item):
            node, anc2 = item
            kids = get_childs(node["model"], node["id"])
            return kids, anc2

        next_frontier = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for kids, anc2 in ex.map(fetch, to_expand):
                for k in kids:
                    next_frontier.append((k, anc2))

        print(f"  level {level}: expanded {len(to_expand)} nodes -> "
              f"{len(next_frontier)} children (leaves so far: {len(leaves)})",
              file=sys.stderr, flush=True)
        frontier = next_frontier

    # build categories map (gom mã lá theo nhóm 3 ký tự, kèm tên node "type")
    categories = {}
    for lf in leaves:
        cat = lf["category"]
        categories.setdefault(cat, {"name": category_names.get(cat, ""), "codes": []})
        categories[cat]["codes"].append(lf["code"])

    out = {
        "meta": {
            "source": "ccs.whiteneuron.com /api/ICD10_TT06 (icd.kcb.vn, Bộ Y tế TT06)",
            "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "leaf_count": len(leaves),
            "category_count": len(categories),
        },
        "codes": leaves,
        "categories": categories,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"DONE: {len(leaves)} leaf codes, {len(categories)} categories "
          f"in {time.time()-t0:.1f}s -> {os.path.relpath(OUT_PATH)}", file=sys.stderr)


if __name__ == "__main__":
    main()
