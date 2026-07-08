def find_position(input_text: str, entity_text: str, used_ranges: list[tuple[int, int]]) -> list[int]:
    start = 0
    while True:
        idx = input_text.find(entity_text, start)
        if idx == -1:
            return []
        end = idx + len(entity_text)
        candidate = (idx, end)
        if candidate not in used_ranges:
            used_ranges.append(candidate)
            return [idx, end]
        start = idx + 1
