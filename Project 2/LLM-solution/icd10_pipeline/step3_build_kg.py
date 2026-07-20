#!/usr/bin/env python3
"""
Convert ICD-10 TT06 flat JSON → Hierarchical JSON → Knowledge Graph.
Input:  data/icd10_tt06.json (18,237 codes from icd.kcb.vn)
Output: data/icd10_hierarchy.json   (hierarchical JSON)
        data/icd10_knowledge_graph.json (nodes + edges for KG/Neo4j)
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
INPUT = os.path.join(DATA_DIR, "icd10_tt06.json")
OUTPUT_HIER = os.path.join(DATA_DIR, "icd10_hierarchy.json")
OUTPUT_KG = os.path.join(DATA_DIR, "icd10_knowledge_graph.json")

CHAPTER_NAMES = {
    "I":    ("Bệnh nhiễm trùng và ký sinh trùng", "Certain infectious and parasitic diseases"),
    "II":   ("U tân sinh", "Neoplasms"),
    "III":  ("Bệnh máu, cơ quan tạo máu và các rối loạn liên quan đến cơ chế miễn dịch",
             "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism"),
    "IV":   ("Bệnh nội tiết, dinh dưỡng và chuyển hóa", "Endocrine, nutritional and metabolic diseases"),
    "V":    ("Rối loạn tâm thần và hành vi", "Mental and behavioural disorders"),
    "VI":   ("Bệnh hệ thần kinh", "Diseases of the nervous system"),
    "VII":  ("Bệnh mắt và phần phụ", "Diseases of the eye and adnexa"),
    "VIII": ("Bệnh của tai và xương chũm", "Diseases of the ear and mastoid process"),
    "IX":   ("Bệnh hệ tuần hoàn", "Diseases of the circulatory system"),
    "X":    ("Bệnh hệ hô hấp", "Diseases of the respiratory system"),
    "XI":   ("Bệnh hệ tiêu hóa", "Diseases of the digestive system"),
    "XII":  ("Bệnh của da và tổ chức dưới da", "Diseases of the skin and subcutaneous tissue"),
    "XIII": ("Bệnh hệ cơ, xương khớp và mô liên kết",
             "Diseases of the musculoskeletal system and connective tissue"),
    "XIV":  ("Bệnh hệ sinh dục, tiết niệu", "Diseases of the genitourinary system"),
    "XV":   ("Thai kỳ, sinh đẻ và hậu sản", "Pregnancy, childbirth and the puerperium"),
    "XVI":  ("Một số bệnh lý khởi phát trong thời kỳ chu sinh",
             "Certain conditions originating in the perinatal period"),
    "XVII": ("Dị tật bẩm sinh, biến dạng và bất thường về nhiễm sắc thể",
             "Congenital malformations, deformations and chromosomal abnormalities"),
    "XVIII":("Triệu chứng, dấu hiệu và những bất thường lâm sàng, cận lâm sàng",
             "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"),
    "XIX":  ("Vết thương, ngộ độc và hậu quả của một số nguyên nhân từ bên ngoài",
             "Injury, poisoning and certain other consequences of external causes"),
    "XX":   ("Các nguyên nhân từ bên ngoài của bệnh tật và tử vong",
             "External causes of morbidity and mortality"),
    "XXI":  ("Các yếu tố liên quan đến tình trạng sức khỏe và tiếp cận dịch vụ y tế",
             "Factors influencing health status and contact with health services"),
    "XXII": ("Mã phục vụ những mục đích đặc biệt", "Codes for special purposes"),
}


def build_hierarchy(codes):
    """Convert flat TT06 codes → hierarchical JSON."""
    # Collect unique entities at each level
    chapters = {}
    sections = {}
    categories = {}

    for c in codes:
        ch = c["chapter"]
        sec = c["section"]
        cat = c["category"]
        disease_code = c["code"]
        disease_name = c["name"]

        if ch not in chapters:
            vi, en = CHAPTER_NAMES.get(ch, ("", ""))
            chapters[ch] = {"code": ch, "name_en": en, "name_vi": vi, "diseases": []}

        sec_key = f"{ch}|{sec}"
        if sec_key not in sections:
            sections[sec_key] = {"code": sec, "chapter": ch, "categories": {}}

        cat_key = f"{sec_key}|{cat}"
        if cat_key not in categories:
            categories[cat_key] = {"code": cat, "section": sec, "chapter": ch, "diseases": []}

        categories[cat_key]["diseases"].append({"code": disease_code, "name_vi": disease_name})

    # Build tree
    chapter_list = []
    for ch in sorted(chapters.keys()):
        vi, en = CHAPTER_NAMES.get(ch, ("", ""))
        ch_sections = [s for s in sections.values() if s["chapter"] == ch]

        section_nodes = []
        for sec in sorted(ch_sections, key=lambda x: x["code"]):
            sec_categories = [c for c in categories.values()
                              if c["section"] == sec["code"] and c["chapter"] == ch]
            cat_nodes = []
            for cat_info in sorted(sec_categories, key=lambda x: x["code"]):
                cat_node = {
                    "code": cat_info["code"],
                    "children": sorted(cat_info["diseases"], key=lambda x: x["code"]),
                    "disease_count": len(cat_info["diseases"]),
                }
                cat_nodes.append(cat_node)

            section_nodes.append({
                "code": sec["code"],
                "children": cat_nodes,
                "category_count": len(cat_nodes),
            })

        chapter_list.append({
            "code": ch,
            "name_en": en,
            "name_vi": vi,
            "children": section_nodes,
            "section_count": len(section_nodes),
        })

    return {"chapters": chapter_list}


def build_knowledge_graph(hierarchy):
    """Convert hierarchical JSON → Knowledge Graph (nodes + relationships)."""
    nodes = []
    edges = []
    node_id = 0

    for ch in hierarchy["chapters"]:
        ch_id = f"ch_{ch['code']}"
        nodes.append({
            "id": ch_id,
            "label": "Chapter",
            "properties": {
                "code": ch["code"],
                "name_vi": ch["name_vi"],
                "name_en": ch["name_en"],
            }
        })

        for sec in ch["children"]:
            sec_id = f"sec_{sec['code']}"
            nodes.append({
                "id": sec_id,
                "label": "Section",
                "properties": {"code": sec["code"]}
            })
            edges.append({"from": ch_id, "to": sec_id, "type": "HAS_CHILD"})

            for cat in sec["children"]:
                cat_id = f"cat_{cat['code']}"
                nodes.append({
                    "id": cat_id,
                    "label": "Category",
                    "properties": {"code": cat["code"]}
                })
                edges.append({"from": sec_id, "to": cat_id, "type": "HAS_CHILD"})

                for dis in cat["children"]:
                    dis_id = f"dis_{dis['code']}"
                    props = {"code": dis["code"], "name_vi": dis.get("name_vi", "")}
                    if "name_en" in dis:
                        props["name_en"] = dis["name_en"]
                    if "code_clean" in dis:
                        props["code_clean"] = dis["code_clean"]
                    nodes.append({"id": dis_id, "label": "Disease", "properties": props})
                    edges.append({"from": cat_id, "to": dis_id, "type": "HAS_CHILD"})

    total_nodes = len(nodes)
    total_edges = len(edges)
    labels = {}
    for n in nodes:
        labels[n["label"]] = labels.get(n["label"], 0) + 1

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "by_label": labels,
        }
    }


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = data["codes"]
    print(f"Loaded {len(codes)} ICD-10 codes from {INPUT}")

    print("Building hierarchical JSON ...")
    hierarchy = build_hierarchy(codes)
    stats = {
        "chapters": len(hierarchy["chapters"]),
        "sections": sum(ch["section_count"] for ch in hierarchy["chapters"]),
        "categories": sum(
            sec["category_count"] for ch in hierarchy["chapters"] for sec in ch["children"]
        ),
        "diseases": sum(
            cat["disease_count"] for ch in hierarchy["chapters"]
            for sec in ch["children"] for cat in sec["children"]
        ),
    }
    hierarchy["stats"] = stats
    print(f"  {stats['chapters']} chapters, {stats['sections']} sections, "
          f"{stats['categories']} categories, {stats['diseases']} diseases")

    with open(OUTPUT_HIER, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {OUTPUT_HIER}")

    print("Building Knowledge Graph ...")
    kg = build_knowledge_graph(hierarchy)
    print(f"  {kg['stats']['total_nodes']} nodes, {kg['stats']['total_edges']} edges")
    print(f"  By label: {kg['stats']['by_label']}")

    with open(OUTPUT_KG, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {OUTPUT_KG}")


if __name__ == "__main__":
    main()
