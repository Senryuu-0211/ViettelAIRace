import json
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    conv_id: int
    turn_idx: int
    arrival_ms: Optional[float]
    in_tokens: float
    out_tokens: float
    think_ms: float
    is_warmup: bool
    in_chars: int
    messages: Optional[list] = field(default=None, repr=False)


@dataclass
class Trace:
    path: str
    fmt: str
    turns: list


def load_trace(path, chars_per_token=3.0):
    with open(path, encoding="utf-8") as f:
        first = json.loads(next(f))
    if "conv_id" in first:
        return _load_grading(path)
    return _load_round1(path, chars_per_token)


def _load_grading(path):
    turns = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            arrival = float(r["timestamp_ms"]) if r["turn_idx"] == 0 else None
            turns.append(Turn(
                conv_id=r["conv_id"],
                turn_idx=r["turn_idx"],
                arrival_ms=arrival,
                in_tokens=float(r["in_tokens_est"]),
                out_tokens=float(r["out_tokens_max"]),
                think_ms=float(r.get("think_ms", 0)),
                is_warmup=bool(r.get("in_warmup", False)),
                in_chars=int(r["in_chars"]),
            ))
    return Trace(path=path, fmt="grading", turns=turns)


def _load_round1(path, chars_per_token):
    conv_keys = {}
    turns = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            msgs = r["body"]["messages"]
            key = msgs[0]["content"][:300] + "\x00" + msgs[1]["content"][:300]
            if key not in conv_keys:
                conv_keys[key] = len(conv_keys)
            conv_id = conv_keys[key]
            turn_idx = (len(msgs) - 2) // 2
            chars = sum(len(m.get("content", "")) for m in msgs)
            turns.append(Turn(
                conv_id=conv_id,
                turn_idx=turn_idx,
                arrival_ms=float(r["timestamp_ms"]),
                in_tokens=chars / chars_per_token,
                out_tokens=float(r["body"].get("max_tokens", 200)),
                think_ms=0.0,
                is_warmup=False,
                in_chars=chars,
                messages=msgs,
            ))
    return Trace(path=path, fmt="round1", turns=turns)


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def basic_stats(vals):
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((v - mean) ** 2 for v in s) / n
    return {
        "n": n,
        "mean": mean,
        "median": percentile(s, 50),
        "std": math.sqrt(var),
        "p50": percentile(s, 50),
        "p90": percentile(s, 90),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
        "min": s[0],
        "max": s[-1],
    }


def ascii_hist(vals, bins=12, width=40):
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return [f"  [{lo:>10,.0f}              ] {'#' * width} {len(vals)}"]
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        i = min(int((v - lo) / step), bins - 1)
        counts[i] += 1
    m = max(counts)
    lines = []
    for i, c in enumerate(counts):
        a = lo + i * step
        b = a + step
        bar = "#" * (round(c / m * width) if c else 0)
        if c and not bar:
            bar = "#"
        lines.append(f"  [{a:>10,.0f} - {b:>10,.0f}) {bar:<{width}} {c}")
    return lines


def group_by_conv(trace):
    convs = {}
    for t in trace.turns:
        convs.setdefault(t.conv_id, []).append(t)
    for v in convs.values():
        v.sort(key=lambda t: t.turn_idx)
    return convs


def fmt_num(x, unit=""):
    if isinstance(x, float) and not x.is_integer():
        return f"{x:,.1f}{unit}"
    return f"{x:,.0f}{unit}"
