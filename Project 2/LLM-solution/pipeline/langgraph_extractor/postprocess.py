from pipeline.config import VALID_TYPES, VALID_ASSERTIONS, CANDIDATE_VALID_TYPES
from pipeline.postprocess.duplicate import expand_duplicates
from pipeline.linker.rxnorm import lookup_drug
from pipeline.linker.icd10 import lookup_diagnosis


def process_and_map_exact_position(raw_text: str, model_outputs: list[dict]) -> list[dict]:
    used_ranges: list[tuple[int, int]] = []

    # 1. Tách raw entities từ LLM, chuẩn hóa type, giữ lại drug field
    raw_entities = []
    for item in model_outputs:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        etype = (item.get("type") or "").strip().upper().replace(" ", "_")
        if etype not in VALID_TYPES:
            continue
        raw_entities.append({
            "text": text,
            "type": etype,
            "quote": (item.get("quote") or text).strip(),
            "assertions": _filter_assertions(item.get("assertions", [])),
            "drug": item.get("drug"),
        })

    # 2. Expand duplicates — clone entity nếu text xuất hiện nhiều lần hơn LLM trả
    entities = expand_duplicates(raw_entities, raw_text)

    # 3. Gán position cho từng entity (dùng quote để neo, fallback text search, chống trùng)
    for e in entities:
        pos = _find_position(raw_text, e["text"], e.get("quote", e["text"]), used_ranges)
        if pos:
            e["position"] = pos
            e["__pos__"] = pos[0]
        else:
            e["position"] = []
            e["__pos__"] = 999999

    entities.sort(key=lambda x: x["__pos__"])

    # 4. Gọi linker cho từng entity (sau khi đã expand + position)
    for e in entities:
        if e["type"] == "THUỐC":
            e["candidates"] = lookup_drug(e["text"], e.get("drug"))
        elif e["type"] == "CHẨN_ĐOÁN":
            e["candidates"] = lookup_diagnosis(e["text"])
        else:
            e["candidates"] = []

    # 5. Strip drug field, validate, dọn key tạm
    result = []
    for e in entities:
        e.pop("drug", None)
        e.pop("quote", None)
        del e["__pos__"]

        _validate_entity(e)
        _strip_candidates_for_non_candidate_types(e)
        result.append(e)

    return result


def _find_position(raw_text: str, entity_text: str, quote: str,
                   used_ranges: list[tuple[int, int]]) -> list[int] | None:
    quote_start = raw_text.find(quote)
    if quote_start != -1:
        text_in_quote = quote.find(entity_text)
        if text_in_quote != -1:
            candidate = (quote_start + text_in_quote, quote_start + text_in_quote + len(entity_text))
            if candidate not in used_ranges:
                used_ranges.append(candidate)
                return [candidate[0], candidate[1]]

    start = 0
    while True:
        idx = raw_text.find(entity_text, start)
        if idx == -1:
            return None
        candidate = (idx, idx + len(entity_text))
        if candidate not in used_ranges:
            used_ranges.append(candidate)
            return [candidate[0], candidate[1]]
        start = idx + 1


def _filter_assertions(assertions) -> list[str]:
    if not isinstance(assertions, list):
        return []
    return [a for a in assertions if isinstance(a, str) and a in VALID_ASSERTIONS]


def _validate_entity(e: dict):
    assert e["type"] in VALID_TYPES, f"Invalid type: {e['type']}"
    assert isinstance(e.get("assertions", []), list), f"Assertions must be list: {e}"
    for a in e.get("assertions", []):
        assert a in VALID_ASSERTIONS, f"Invalid assertion: {a}"
    for field in ("text", "type", "position", "assertions", "candidates"):
        assert field in e, f"Missing required field: {field}"


def _strip_candidates_for_non_candidate_types(e: dict):
    if e["type"] not in CANDIDATE_VALID_TYPES:
        e.pop("candidates", None)
