import json
import re
import time

from pipeline.chains.entity_chain import get_entity_chain
from pipeline.config import MAX_RETRIES, VALID_TYPES


def extract_entities(input_text: str) -> list[dict]:
    chain = get_entity_chain()

    for attempt in range(MAX_RETRIES):
        try:
            raw = chain.invoke({"input_text": input_text})
            entities = _parse_entities(raw, input_text)
            if entities:
                return entities
        except Exception as e:
            wait = min(5 * (2 ** attempt), 30)
            print(f"  [RETRY {attempt+1}/{MAX_RETRIES}] Entity extraction error: {e}")
            time.sleep(wait)

    return []


def extract_entities_batch(texts: list[str]) -> list[list[dict]]:
    chain = get_entity_chain()
    inputs = [{"input_text": t} for t in texts]

    for attempt in range(MAX_RETRIES):
        try:
            raws = chain.batch(inputs, config={"max_concurrency": len(texts)})
            results = []
            for raw, text in zip(raws, texts):
                entities = _parse_entities(raw, text)
                results.append(entities if entities else [])
            return results
        except Exception as e:
            wait = min(5 * (2 ** attempt), 30)
            print(f"  [RETRY {attempt+1}/{MAX_RETRIES}] Batch entity extraction error: {e}")
            time.sleep(wait)

    return [[] for _ in texts]


def _parse_entities(raw: str, input_text: str) -> list[dict]:
    raw = raw.strip()
    for prefix in ("```json", "```"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    for suffix in ("```",):
        if raw.endswith(suffix):
            raw = raw[:-len(suffix)]
    raw = raw.strip()

    result = []
    decoder = json.JSONDecoder()
    pos = 0

    while pos < len(raw):
        try:
            obj, new_pos = decoder.raw_decode(raw, pos)
            items = obj if isinstance(obj, list) else [obj] if isinstance(obj, dict) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = (item.get("text") or "").strip()
                etype = (item.get("type") or "").strip().upper().replace(" ", "_")
                if text and etype in VALID_TYPES:
                    if text in input_text:
                        result.append({"text": text, "type": etype, "drug": item.get("drug")})
            pos = new_pos
        except json.JSONDecodeError:
            pos += 1

    if not result:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    text = (item.get("text") or "").strip()
                    etype = (item.get("type") or "").strip().upper().replace(" ", "_")
                    if not text or etype not in VALID_TYPES:
                        continue
                    if text not in input_text:
                        continue
                    result.append({"text": text, "type": etype, "drug": item.get("drug")})

    return result
