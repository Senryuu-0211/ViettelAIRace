import argparse
import math

from trace_common import load_trace
from simulate_scheduler import simulate

MODEL_PRESETS = {
    "lfm2.5-1.2b": {
        "label": "LiquidAI/LFM2.5-1.2B-Instruct (hybrid conv+attention)",
        "params_b": 1.2,
        "attn_layers": 6,
        "conv_layers": 10,
        "kv_heads": 8,
        "head_dim": 64,
        "hidden": 2048,
    },
    "qwen3.5-2b": {
        "label": "Qwen3.5-2B (assumed dense transformer)",
        "params_b": 2.0,
        "attn_layers": 28,
        "conv_layers": 0,
        "kv_heads": 8,
        "head_dim": 128,
        "hidden": 2048,
    },
}

DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}


def kv_bytes_per_token(cfg, kv_dtype):
    return 2 * cfg["attn_layers"] * cfg["kv_heads"] * cfg["head_dim"] * DTYPE_BYTES[kv_dtype]


def round_up(x, mult):
    return int(math.ceil(x / mult) * mult)


def round_up_pow2(x):
    return 1 << max(0, math.ceil(math.log2(max(x, 1))))


def gather_workload_stats(trace):
    in_toks = [t.in_tokens for t in trace.turns]
    out_toks = [t.out_tokens for t in trace.turns]
    ctx = [i + o for i, o in zip(in_toks, out_toks)]
    s = sorted(ctx)
    return {
        "max_in": max(in_toks),
        "max_out": max(out_toks),
        "max_ctx": max(ctx),
        "p95_ctx": s[min(len(s) - 1, int(0.95 * len(s)))],
        "n_requests": len(trace.turns),
    }


def print_memory(ws, sim, cfg, args):
    kv_tok = kv_bytes_per_token(cfg, args.kv_dtype)
    weights_gb = cfg["params_b"] * DTYPE_BYTES[args.weights_dtype]
    budget_gb = args.gpu_mem_gb * args.gpu_util
    overhead_gb = args.overhead_gb
    kv_budget_gb = budget_gb - weights_gb - overhead_gb
    capacity_tokens = kv_budget_gb * 1024 ** 3 / kv_tok if kv_budget_gb > 0 else 0
    peak_active = sim.peak_concurrency * ws["max_ctx"]
    peak_active_gb = peak_active * kv_tok / 1024 ** 3

    print("================ MEMORY ================")
    print(f"Model preset            : {cfg['label']}")
    print(f"KV bytes/token          : {kv_tok:,} B "
          f"({cfg['attn_layers']} attn layers x {cfg['kv_heads']} kv_heads x "
          f"{cfg['head_dim']} head_dim, {args.kv_dtype})")
    if cfg["conv_layers"]:
        print(f"Note                    : {cfg['conv_layers']} conv layers use tiny "
              f"constant per-seq state (not KV) -> KV cache is small vs dense models")
    print(f"GPU memory              : {args.gpu_mem_gb:.1f} GB "
          f"x util {args.gpu_util} = {budget_gb:.2f} GB usable")
    print(f"Weights ({args.weights_dtype})          : {weights_gb:.2f} GB")
    print(f"Runtime overhead est.   : {overhead_gb:.2f} GB (activations, CUDA graphs, buffers)")
    print(f"KV cache budget         : {kv_budget_gb:.2f} GB "
          f"-> capacity ~{capacity_tokens:,.0f} tokens")
    print(f"Peak concurrency (sim)  : {sim.peak_concurrency}")
    print(f"Peak active tokens      : {sim.peak_concurrency} seqs x {ws['max_ctx']:,.0f} ctx "
          f"= {peak_active:,.0f} tokens")
    print(f"Estimated KV at peak    : {peak_active_gb:.3f} GB "
          f"({peak_active / max(capacity_tokens, 1) * 100:.1f}% of KV budget)")
    if peak_active > capacity_tokens:
        print("WARNING                 : peak KV demand exceeds budget -> "
              "consider --kv-cache-dtype fp8, lower max-num-seqs, or lower max-model-len")
    print()
    return capacity_tokens


def print_recommendation(ws, sim, cfg, args, capacity_tokens):
    min_len = ws["max_in"] + ws["max_out"]
    rec_len = max(round_up(min_len * 1.1, 1024), 4096)
    peak = max(sim.peak_concurrency, 1)
    rec_seqs = max(round_up(peak * 1.5, 8), 16)
    if capacity_tokens > 0:
        cap_seqs = int(capacity_tokens // ws["max_ctx"])
        rec_seqs = min(rec_seqs, max(cap_seqs, 8))
    rec_batched = max(round_up_pow2(ws["max_in"]), 2048)
    prefill_ms = rec_batched / args.prefill_tps * 1000

    print("================ RECOMMENDATION ================")
    print(f"--max-model-len {rec_len}")
    print(f"    max prompt+output in trace = {min_len:,.0f} tokens; "
          f"{rec_len} gives headroom without wasting KV blocks")
    print(f"--max-num-seqs {rec_seqs}")
    print(f"    simulated peak concurrency = {peak}; x1.5 headroom, "
          f"capped by KV capacity ({capacity_tokens:,.0f} tokens)")
    print(f"--max-num-batched-tokens {rec_batched}")
    print(f"    >= max prompt ({ws['max_in']:,.0f} tok) so a full prompt prefills in one step; "
          f"~{prefill_ms:.0f} ms at {args.prefill_tps:,.0f} tok/s "
          f"(TTFT ceiling {400} ms)")
    print(f"--enable-prefix-caching")
    print(f"    multi-turn workload: each turn re-sends conversation history")
    print(f"--gpu-memory-utilization {args.gpu_util}")
    if capacity_tokens < peak * ws["max_ctx"] * 2:
        print(f"--kv-cache-dtype fp8")
        print(f"    doubles KV capacity; small accuracy cost (check GPQA gate)")
    print()
    print("Suggested vLLM command line:")
    flags = [
        f"--max-model-len={rec_len}",
        f"--max-num-seqs={rec_seqs}",
        f"--max-num-batched-tokens={rec_batched}",
        "--enable-prefix-caching",
        f"--gpu-memory-utilization={args.gpu_util}",
        "--tensor-parallel-size=1",
    ]
    print("  " + " \\\n  ".join(flags))


def build_argparser(ap=None):
    ap = ap or argparse.ArgumentParser(description="Recommend vLLM config from a trace")
    ap.add_argument("--model", choices=sorted(MODEL_PRESETS), default="lfm2.5-1.2b")
    ap.add_argument("--gpu-mem-gb", type=float, default=18.0)
    ap.add_argument("--gpu-util", type=float, default=0.9)
    ap.add_argument("--overhead-gb", type=float, default=1.5)
    ap.add_argument("--weights-dtype", choices=sorted(DTYPE_BYTES), default="bf16")
    ap.add_argument("--kv-dtype", choices=sorted(DTYPE_BYTES), default="bf16")
    ap.add_argument("--prefill-tps", type=float, default=10000.0)
    ap.add_argument("--tpot-ms", type=float, default=8.0)
    ap.add_argument("--max-num-seqs-sim", type=int, default=256)
    ap.add_argument("--prefix-hit", type=float, default=0.5)
    ap.add_argument("--chars-per-token", type=float, default=3.0)
    return ap


def run(trace, args):
    cfg = MODEL_PRESETS[args.model]
    ws = gather_workload_stats(trace)
    sim = simulate(trace, prefill_tps=args.prefill_tps, tpot_ms=args.tpot_ms,
                   max_num_seqs=args.max_num_seqs_sim, prefix_hit=args.prefix_hit)
    capacity = print_memory(ws, sim, cfg, args)
    print_recommendation(ws, sim, cfg, args, capacity)


def main():
    ap = build_argparser()
    ap.add_argument("trace")
    args = ap.parse_args()
    trace = load_trace(args.trace, args.chars_per_token)
    run(trace, args)


if __name__ == "__main__":
    main()
