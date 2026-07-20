import json
import re
import time

from pipeline.chains.assertion_chain import get_assertion_chain
from pipeline.config import MAX_RETRIES, MAX_ASSERTION_BATCH, ASSERTION_VALID_TYPES, VALID_ASSERTIONS


def extract_assertions_batch(input_text: str, entities: list[dict]) -> dict[str, list[str]]:
    eligible = [e for e in entities if e["type"] in ASSERTION_VALID_TYPES]
    if not eligible:
        return {}

    chunks = [eligible[i:i + MAX_ASSERTION_BATCH] for i in range(0, len(eligible), MAX_ASSERTION_BATCH)]
    result = {}

    for chunk in chunks:
        entity_list = "\n".join(f"- {e['text']} ({e['type']})" for e in chunk)
        chunk_result = _call_assertion(input_text, entity_list)
        result.update(chunk_result)

    return result


def _call_assertion(input_text: str, entity_list: str) -> dict[str, list[str]]:
    chain = get_assertion_chain()

    for attempt in range(MAX_RETRIES):
        try:
            raw = chain.invoke({"input_text": input_text, "entity_list": entity_list})
            parsed = _parse_batch_assertions(raw)
            if parsed:
                return parsed
        except Exception as e:
            wait = min(5 * (2 ** attempt), 30)
            print(f"  [RETRY {attempt+1}/{MAX_RETRIES}] Batch assertion error: {e}")
            time.sleep(wait)

    return {}


def _parse_batch_assertions(raw: str) -> dict[str, list[str]]:
    raw = raw.strip()
    for prefix in ("```json", "```"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    for suffix in ("```",):
        if raw.endswith(suffix):
            raw = raw[:-len(suffix)]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    if not isinstance(data, dict):
        return {}

    result = {}
    for key, val in data.items():
        if isinstance(val, list):
            if val and isinstance(val[0], dict) and "text" in val[0]:
                for item in val:
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"]
                        assertions = _extract_assertions(item.get("assertions", []))
                        if text:
                            result[text] = assertions
            else:
                result[key] = _extract_assertions(val)
        elif isinstance(val, dict) and "text" in val:
            text = val["text"]
            assertions = _extract_assertions(val.get("assertions", []))
            if text:
                result[text] = assertions
        elif isinstance(val, str):
            result[key] = [val] if val in VALID_ASSERTIONS else []
        else:
            result[key] = []
    return result


def _extract_assertions(val: list) -> list[str]:
    res = []
    for v in val:
        if isinstance(v, str):
            res.append(v)
        elif isinstance(v, dict):
            res.extend(x for x in v.values() if isinstance(x, str))
    return [a for a in res if a in VALID_ASSERTIONS]
