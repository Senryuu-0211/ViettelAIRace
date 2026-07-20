import argparse
from collections import defaultdict

from trace_common import ascii_hist, basic_stats, group_by_conv, load_trace
from simulate_scheduler import simulate
import recommend_vllm


def section(title):
    print(f"\n================ {title} ================\n")


def print_stats_block(vals, unit="tokens"):
    st = basic_stats(vals)
    print(f"Mean    : {st['mean']:,.1f} {unit}")
    print(f"Median  : {st['median']:,.1f} {unit}")
    print(f"Std     : {st['std']:,.1f} {unit}")
    print()
    print(f"P50     : {st['p50']:,.1f}")
    print(f"P90     : {st['p90']:,.1f}")
    print(f"P95     : {st['p95']:,.1f}")
    print(f"P99     : {st['p99']:,.1f}")
    print()
    print(f"Min     : {st['min']:,.1f}")
    print(f"Max     : {st['max']:,.1f}")
    print()
    print("Histogram:")
    for line in ascii_hist(vals):
        print(line)


def prompt_text(turn):
    return "\x00".join(f"{m['role']}:{m.get('content', '')}" for m in turn.messages)


def prefix_reuse_round1(convs):
    shared_total = 0
    len_total = 0
    ratios = []
    for ts in convs.values():
        prev = None
        for t in ts:
            cur = prompt_text(t)
            if prev is not None:
                n = min(len(prev), len(cur))
                i = 0
                while i < n and prev[i] == cur[i]:
                    i += 1
                shared_total += i
                len_total += len(cur)
                ratios.append(i / len(cur))
            prev = cur
    overall = shared_total / len_total if len_total else 0.0
    return overall, ratios


def analyze(trace, args):
    convs = group_by_conv(trace)
    turns = trace.turns
    warmup = [t for t in turns if t.is_warmup]
    graded = [t for t in turns if not t.is_warmup]

    section("DATASET")
    print(f"File               : {trace.path}")
    print(f"Format             : {trace.fmt}")
    print(f"Total conversations: {len(convs)}")
    print(f"Total requests     : {len(turns)}")
    print()
    print(f"Warmup requests    : {len(warmup)}")
    print()
    print(f"Grading requests   : {len(graded)}")

    section("INPUT TOKENS")
    if trace.fmt == "round1":
        print(f"(estimated from chars / {args.chars_per_token} chars-per-token; "
              f"full prompt = system + history + new user)")
    print_stats_block([t.in_tokens for t in turns])

    section("OUTPUT TOKENS")
    print("(from max_tokens / out_tokens_max -> upper bound, not actual generation)")
    print_stats_block([t.out_tokens for t in turns])

    section("CONVERSATION")
    n_turns = [len(v) for v in convs.values()]
    print(f"Average turns      : {sum(n_turns) / len(n_turns):.2f} "
          f"(min {min(n_turns)}, max {max(n_turns)})")
    print()
    by_idx = defaultdict(list)
    for t in turns:
        by_idx[t.turn_idx].append(t.in_tokens)
    print("Prompt growth (avg input tokens per turn_idx):")
    prev_avg = None
    for idx in sorted(by_idx):
        avg = sum(by_idx[idx]) / len(by_idx[idx])
        delta = f"  ({avg - prev_avg:+,.1f})" if prev_avg is not None else ""
        print(f"  turn {idx}: {avg:>12,.1f}{delta}")
        prev_avg = avg
    print()
    first_idx, last_idx = min(by_idx), max(by_idx)
    a0 = sum(by_idx[first_idx]) / len(by_idx[first_idx])
    a1 = sum(by_idx[last_idx]) / len(by_idx[last_idx])
    span = max(last_idx - first_idx, 1)
    print(f"History growth     : {(a1 - a0) / span:,.1f} tokens/turn "
          f"({a0:,.0f} -> {a1:,.0f} over {span} turns)")
    print()
    if trace.fmt == "round1" and turns[0].messages is not None:
        overall, ratios = prefix_reuse_round1(convs)
        st = basic_stats(ratios)
        print(f"Prefix reuse estimation (measured, chars shared with previous turn prompt):")
        print(f"  overall = {overall * 100:.1f}%   per-turn mean = {st['mean'] * 100:.1f}%  "
              f"min = {st['min'] * 100:.1f}%  max = {st['max'] * 100:.1f}%")
    else:
        print("Prefix reuse estimation: N/A (public trace has no prompt text).")
        print("  Per-turn prompt size is ~constant (~4K tok) despite multi-turn ->")
        print("  history likely trimmed/rolling window; assume moderate reuse (50-80%),")
        print("  still worth --enable-prefix-caching.")

    section("ARRIVAL")
    explicit = sorted(t.arrival_ms for t in turns if t.arrival_ms is not None)
    label = "request" if trace.fmt == "round1" else "conversation (turn 0)"
    ia = [b - a for a, b in zip(explicit, explicit[1:])]
    if ia:
        st = basic_stats(ia)
        print(f"Inter-arrival ({label} level):")
        print(f"  mean = {st['mean']:,.1f} ms  median = {st['median']:,.1f} ms  "
              f"std = {st['std']:,.1f} ms")
        print(f"  min  = {st['min']:,.1f} ms  p95 = {st['p95']:,.1f} ms  "
              f"max = {st['max']:,.1f} ms")
        rate = len(explicit) / max(explicit[-1] - explicit[0], 1) * 1000
        print(f"  arrival rate ~ {rate:.2f} {label.split()[0]}s/s over "
              f"{(explicit[-1] - explicit[0]) / 1000:.1f} s window")
    print()
    sim = simulate(trace, prefill_tps=args.prefill_tps, tpot_ms=args.tpot_ms,
                   max_num_seqs=10 ** 6, prefix_hit=args.prefix_hit)
    print(f"(concurrency below from event simulation: prefill_tps={args.prefill_tps:,.0f}, "
          f"tpot={args.tpot_ms}ms, prefix_hit={args.prefix_hit}, no seq limit)")
    print(f"Peak concurrency   : {sim.peak_concurrency}")
    print()
    print(f"Average concurrency: {sim.avg_concurrency:.2f}")
    print()
    print(f"Idle time          : {sim.idle_ms / 1000:.1f} s "
          f"({sim.idle_ms / max(sim.makespan_ms, 1) * 100:.1f}% of "
          f"{sim.makespan_ms / 1000:.1f} s makespan)")
    print()

    rec_args = recommend_vllm.build_argparser().parse_args([])
    rec_args.model = args.model
    rec_args.prefill_tps = args.prefill_tps
    rec_args.tpot_ms = args.tpot_ms
    rec_args.prefix_hit = args.prefix_hit
    recommend_vllm.run(trace, rec_args)


def main():
    ap = argparse.ArgumentParser(description="Offline statistics for multi-turn LLM traces")
    ap.add_argument("trace")
    ap.add_argument("--chars-per-token", type=float, default=3.0)
    ap.add_argument("--prefill-tps", type=float, default=10000.0)
    ap.add_argument("--tpot-ms", type=float, default=8.0)
    ap.add_argument("--prefix-hit", type=float, default=0.5)
    ap.add_argument("--model", choices=sorted(recommend_vllm.MODEL_PRESETS),
                    default="lfm2.5-1.2b")
    args = ap.parse_args()
    trace = load_trace(args.trace, args.chars_per_token)
    analyze(trace, args)


if __name__ == "__main__":
    main()
