import re


def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def verify_text_in_input(entity_text: str, input_text: str) -> bool:
    return entity_text in input_text
