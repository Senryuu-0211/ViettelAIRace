import json
import re
import time

from pipeline.chains.assertion_chain import get_assertion_chain
from pipeline.config import MAX_RETRIES, ASSERTION_VALID_TYPES, VALID_ASSERTIONS


def extract_assertions_batch(input_text: str, entities: list[dict]) -> dict[str, list[str]]:
    eligible = [e for e in entities if e["type"] in ASSERTION_VALID_TYPES]
    if not eligible:
        return {}

    chain = get_assertion_chain()
    entity_list = "\n".join(f"- {e['text']} ({e['type']})" for e in eligible)

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
            result[key] = [a for a in val if a in VALID_ASSERTIONS]
        else:
            result[key] = []
    return result
