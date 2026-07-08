def expand_duplicates(entities: list[dict], input_text: str) -> list[dict]:
    result = []
    seen_count: dict[str, int] = {}

    for e in entities:
        text = e["text"]
        total_occurrences = _count_occurrences(input_text, text)
        current_count = seen_count.get(text, 0)
        missing = total_occurrences - current_count

        for _ in range(missing):
            result.append(dict(e))
            seen_count[text] = seen_count.get(text, 0) + 1

    return result


def _count_occurrences(text: str, substring: str) -> int:
    if not substring:
        return 0
    count = 0
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx == -1:
            break
        count += 1
        start = idx + 1
    return count
