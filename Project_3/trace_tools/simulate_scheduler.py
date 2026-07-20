import argparse
import heapq
from collections import deque
from dataclasses import dataclass

from trace_common import Trace, basic_stats, group_by_conv, load_trace, percentile

ERS_F_TTFT = 10.0
ERS_C_TTFT = 400.0
ERS_F_TPOT = 1.0
ERS_C_TPOT = 10.0
ERS_GAMMA = 2.0
ERS_W = 0.5


@dataclass
class ReqResult:
    conv_id: int
    turn_idx: int
    is_warmup: bool
    arrival_ms: float
    start_ms: float
    finish_ms: float
    ttft_ms: float
    tpot_ms: float


@dataclass
class SimResult:
    results: list
    peak_concurrency: int
    avg_concurrency: float
    idle_ms: float
    makespan_ms: float
    conc_events: list


def ers_component(x, floor, ceil, gamma=ERS_GAMMA):
    v = (ceil - x) / (ceil - floor)
    v = max(0.0, min(1.0, v))
    return v ** gamma


def request_score(ttft_ms, tpot_ms):
    s_ttft = ers_component(ttft_ms, ERS_F_TTFT, ERS_C_TTFT)
    s_tpot = ers_component(tpot_ms, ERS_F_TPOT, ERS_C_TPOT)
    return ERS_W * s_ttft + (1 - ERS_W) * s_tpot, s_ttft, s_tpot


def simulate(trace, prefill_tps=10000.0, tpot_ms=8.0, max_num_seqs=32,
             prefix_hit=0.5, turn0_hit=0.0, think_ms_default=3000.0):
    convs = group_by_conv(trace)
    heap = []
    seq = 0
    for cid, ts in convs.items():
        for i, t in enumerate(ts):
            if t.arrival_ms is not None:
                heapq.heappush(heap, (t.arrival_ms, seq, cid, i))
                seq += 1

    queue = deque()
    running = 0
    results = []
    conc_events = []

    def service_ms(turn):
        hit = turn0_hit if turn.turn_idx == 0 else prefix_hit
        prefill = turn.in_tokens * (1.0 - hit) / prefill_tps * 1000.0
        decode = turn.out_tokens * tpot_ms
        return prefill, decode

    finish_heap = []

    def start_pending(now):
        nonlocal running, seq
        while queue and running < max_num_seqs:
            arr, cid, idx = queue.popleft()
            turn = convs[cid][idx]
            prefill, decode = service_ms(turn)
            ttft = (now - arr) + prefill
            finish = now + prefill + decode
            running += 1
            conc_events.append((now, 1))
            results.append(ReqResult(cid, turn.turn_idx, turn.is_warmup,
                                     arr, now, finish, ttft, tpot_ms))
            heapq.heappush(finish_heap, (finish, seq, cid, idx))
            seq += 1

    while heap or finish_heap:
        if finish_heap and (not heap or finish_heap[0][0] <= heap[0][0]):
            t, _, cid, idx = heapq.heappop(finish_heap)
            running -= 1
            conc_events.append((t, -1))
            nxt = idx + 1
            if nxt < len(convs[cid]) and convs[cid][nxt].arrival_ms is None:
                think = convs[cid][nxt].think_ms or think_ms_default
                heapq.heappush(heap, (t + think, seq, cid, nxt))
                seq += 1
            start_pending(t)
        else:
            t, _, cid, idx = heapq.heappop(heap)
            queue.append((t, cid, idx))
            start_pending(t)

    conc_events.sort()
    peak = 0
    cur = 0
    avg_num = 0.0
    idle = 0.0
    prev_t = conc_events[0][0] if conc_events else 0.0
    t0 = prev_t
    for t, d in conc_events:
        dt = t - prev_t
        avg_num += cur * dt
        if cur == 0:
            idle += dt
        cur += d
        peak = max(peak, cur)
        prev_t = t
    makespan = prev_t - t0 if conc_events else 0.0
    avg = avg_num / makespan if makespan > 0 else 0.0
    return SimResult(results, peak, avg, idle, prev_t, conc_events)


def concurrency_timeline(sim, buckets=60, width=50):
    if not sim.conc_events:
        return []
    t0 = sim.conc_events[0][0]
    t1 = sim.makespan_ms
    span = max(t1 - t0, 1.0)
    step = span / buckets
    acc = [0.0] * buckets
    cur = 0
    prev = t0
    for t, d in sim.conc_events:
        a, b = prev, t
        if b > a and cur > 0:
            i0 = int((a - t0) / step)
            i1 = int((b - t0) / step)
            for i in range(i0, min(i1 + 1, buckets)):
                lo = t0 + i * step
                hi = lo + step
                ov = max(0.0, min(b, hi) - max(a, lo))
                acc[i] += cur * ov
        cur += d
        prev = t
    avgs = [a / step for a in acc]
    m = max(avgs) or 1.0
    lines = []
    for i, v in enumerate(avgs):
        bar = "#" * round(v / m * width)
        lines.append(f"  {(t0 + i * step) / 1000:>7.1f}s |{bar:<{width}}| {v:.1f}")
    return lines


def print_report(trace, sim, args):
    graded = [r for r in sim.results if not r.is_warmup]
    scored = []
    for r in graded:
        s, st, sp = request_score(r.ttft_ms, r.tpot_ms)
        scored.append((s, st, sp))
    ttfts = sorted(r.ttft_ms for r in graded)
    tpots = sorted(r.tpot_ms for r in graded)

    print("================ SIMULATION CONFIG ================")
    print(f"Trace                 : {trace.path} ({trace.fmt})")
    print(f"prefill_tps           : {args.prefill_tps:,.0f} tok/s")
    print(f"tpot_ms               : {args.tpot_ms} ms")
    print(f"max_num_seqs          : {args.max_num_seqs}")
    print(f"prefix_hit (turn>0)   : {args.prefix_hit}")
    print(f"turn0_hit             : {args.turn0_hit}")
    print()
    print("================ CONCURRENCY ================")
    print(f"Peak concurrency      : {sim.peak_concurrency}")
    print(f"Average concurrency   : {sim.avg_concurrency:.2f}")
    print(f"Makespan              : {sim.makespan_ms / 1000:.1f} s")
    print(f"Idle time             : {sim.idle_ms / 1000:.1f} s "
          f"({sim.idle_ms / max(sim.makespan_ms, 1) * 100:.1f}%)")
    print()
    print("Timeline (avg concurrency per bucket):")
    for line in concurrency_timeline(sim):
        print(line)
    print()
    print("================ LATENCY (graded requests) ================")
    st = basic_stats(ttfts)
    print(f"TTFT  mean={st['mean']:.1f}ms p50={st['p50']:.1f} p90={st['p90']:.1f} "
          f"p95={st['p95']:.1f} p99={st['p99']:.1f} max={st['max']:.1f}")
    sp = basic_stats(tpots)
    print(f"TPOT  mean={sp['mean']:.2f}ms p95={sp['p95']:.2f} max={sp['max']:.2f}")
    over = sum(1 for t in ttfts if t > ERS_C_TTFT)
    print(f"Requests with TTFT > {ERS_C_TTFT:.0f}ms ceiling: {over}/{len(ttfts)}")
    print()
    print("================ ERS ESTIMATE ================")
    ers = sum(s for s, _, _ in scored) / len(scored) if scored else 0.0
    mst = sum(st for _, st, _ in scored) / len(scored) if scored else 0.0
    msp = sum(sp for _, _, sp in scored) / len(scored) if scored else 0.0
    print(f"mean s_ttft           : {mst:.4f}")
    print(f"mean s_tpot           : {msp:.4f}")
    print(f"ERS                   : {ers:.4f}")
    print(f"Score (x100, f=1)     : {ers * 100:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Event-driven concurrency simulator for multi-turn traces")
    ap.add_argument("trace")
    ap.add_argument("--prefill-tps", type=float, default=10000.0)
    ap.add_argument("--tpot-ms", type=float, default=8.0)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--prefix-hit", type=float, default=0.5)
    ap.add_argument("--turn0-hit", type=float, default=0.0)
    ap.add_argument("--think-ms", type=float, default=3000.0)
    ap.add_argument("--chars-per-token", type=float, default=3.0)
    ap.add_argument("--sweep-seqs", type=str, default="")
    args = ap.parse_args()

    trace = load_trace(args.trace, args.chars_per_token)
    if args.sweep_seqs:
        print(f"{'max_num_seqs':>12} | {'peak':>5} | {'avg':>6} | {'TTFT p95':>9} | {'ERS':>6}")
        print("-" * 55)
        for s in [int(x) for x in args.sweep_seqs.split(",")]:
            sim = simulate(trace, args.prefill_tps, args.tpot_ms, s,
                           args.prefix_hit, args.turn0_hit, args.think_ms)
            graded = [r for r in sim.results if not r.is_warmup]
            ttfts = sorted(r.ttft_ms for r in graded)
            ers = sum(request_score(r.ttft_ms, r.tpot_ms)[0] for r in graded) / len(graded)
            print(f"{s:>12} | {sim.peak_concurrency:>5} | {sim.avg_concurrency:>6.2f} | "
                  f"{percentile(ttfts, 95):>8.1f}ms | {ers:>6.4f}")
        return

    sim = simulate(trace, args.prefill_tps, args.tpot_ms, args.max_num_seqs,
                   args.prefix_hit, args.turn0_hit, args.think_ms)
    print_report(trace, sim, args)


if __name__ == "__main__":
    main()
