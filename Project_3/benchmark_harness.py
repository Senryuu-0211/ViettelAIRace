"""
Benchmark harness: replay trace-round1.jsonl theo đúng arrival timestamp,
gọi vào endpoint OpenAI-compatible của vLLM, đo TTFT/TPOT từng request,
và tính ERS cục bộ theo đúng công thức chấm điểm Phase 1.

Cách dùng:
    pip install aiohttp --break-system-packages
    python benchmark_harness.py trace-round1.jsonl --url http://localhost:8000/v1 --model Qwen3.5-2B

Lưu ý: script này chỉ đo phần ERS (latency). Accuracy Gate (GPQA Diamond)
cần chạy riêng bằng eval harness (vd lm-evaluation-harness) trỏ vào cùng
endpoint, rồi tự tính f(delta) theo công thức piecewise linear.
"""
import json
import asyncio
import aiohttp
import time
import argparse
import statistics

# --- Tham số chấm điểm Phase 1 (bảng công bố trong đề bài) ---
F_TTFT, C_TTFT = 0.100, 1.500   # giây
F_TPOT, C_TPOT = 0.020, 0.045   # giây
GAMMA = 2
W = 0.5


def clamp(x, lo=0.0, hi=1.0):

    return max(lo, min(hi, x))


def score_component(value, floor, ceiling, gamma):
    x = clamp((ceiling - value) / (ceiling - floor))
    return x ** gamma


def request_score(ttft, tpot_mean, success):
    if not success:
        return 0.0
    s_ttft = score_component(ttft, F_TTFT, C_TTFT, GAMMA)
    s_tpot = score_component(tpot_mean, F_TPOT, C_TPOT, GAMMA)
    return W * s_ttft + (1 - W) * s_tpot


def extract_request_data(record):
    body = record.get("body", {})
    messages = body.get("messages", []) if isinstance(body, dict) else []
    prompt = record.get("prompt") or record.get("input") or ""
    if not prompt and messages:
        prompt = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))
    max_tokens = record.get("max_tokens") or (isinstance(body, dict) and body.get("max_tokens")) or 100
    return prompt, max_tokens, messages


async def send_request(session, url, model, record, results, idx):
    prompt, max_tokens, messages = extract_request_data(record)

    if messages:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
        }
        endpoint = f"{url}/chat/completions"
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
        }
        endpoint = f"{url}/completions"

    t_start = time.perf_counter()
    ttft = None
    token_times = []
    got_text = False

    try:
        async with session.post(
            endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t_start
                else:
                    token_times.append(now)
                try:
                    chunk = json.loads(data_str)
                    choice = chunk["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        text = choice["delta"]["content"]
                    else:
                        text = choice.get("text", "")
                    if text:
                        got_text = True
                except Exception:
                    pass
    except Exception as e:
        results[idx] = {"idx": idx, "success": False, "error": str(e)}
        return

    if ttft is None or not got_text:
        results[idx] = {"idx": idx, "success": False, "error": "no_token_received"}
        return

    # TPOT = khoảng cách trung bình giữa các token SAU token đầu tiên
    if len(token_times) >= 2:
        gaps = [t2 - t1 for t1, t2 in zip(token_times, token_times[1:])]
        tpot_mean = statistics.mean(gaps)
    else:
        tpot_mean = 0.0  # chỉ 1 token output -> không có TPOT để phạt

    results[idx] = {
        "idx": idx,
        "success": True,
        "ttft": ttft,
        "tpot": tpot_mean,
        "score": request_score(ttft, tpot_mean, True),
    }


async def run_trace(trace_path, url, model, concurrency_limit=None):
    with open(trace_path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    n = len(records)
    results = [None] * n

    raw_ts = [
        r.get("timestamp_ms", r.get("timestamp", r.get("arrival_time", r.get("ts", i * 0.1))))
        for i, r in enumerate(records)
    ]
    t0 = min(raw_ts)
    arrival_offsets = [(float(t) - float(t0)) / 1000.0 for t in raw_ts]

    sem = asyncio.Semaphore(concurrency_limit) if concurrency_limit else None

    async def scheduled_send(session, idx, record, delay):
        await asyncio.sleep(delay)
        if sem:
            async with sem:
                await send_request(session, url, model, record, results, idx)
        else:
            await send_request(session, url, model, record, results, idx)

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(scheduled_send(session, i, r, arrival_offsets[i]))
            for i, r in enumerate(records)
        ]
        await asyncio.gather(*tasks)

    return results


def summarize(results):
    n = len(results)
    scores = [r["score"] if r and r.get("score") is not None else 0.0 for r in results]
    ers = statistics.mean(scores) if scores else 0.0
    n_fail = sum(1 for r in results if not r or not r.get("success"))

    print("\n=== Kết quả benchmark local ===")
    print(f"Tổng request: {n} | Thất bại/timeout: {n_fail}")
    print(f"ERS (local) = {ers:.4f}")

    ttfts = sorted(r["ttft"] for r in results if r and r.get("success"))
    tpots = sorted(r["tpot"] for r in results if r and r.get("success"))
    if ttfts:
        print(f"TTFT: mean={statistics.mean(ttfts)*1000:.1f}ms, "
              f"p90={ttfts[int(len(ttfts)*0.9)]*1000:.1f}ms")
    if tpots:
        print(f"TPOT: mean={statistics.mean(tpots)*1000:.1f}ms, "
              f"p90={tpots[int(len(tpots)*0.9)]*1000:.1f}ms")

    print("\n(Score cuối = 100 x ERS x f(delta) -- chạy riêng eval GPQA Diamond "
          "để lấy f(delta), script này chỉ đo phần ERS/latency.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_path")
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen3.5-2B")
    ap.add_argument(
        "--concurrency-limit", type=int, default=None,
        help="Giới hạn số request gửi đồng thời từ CLIENT (không phải server), "
             "để tránh chính client trở thành bottleneck."
    )
    args = ap.parse_args()

    results = asyncio.run(
        run_trace(args.trace_path, args.url, args.model, args.concurrency_limit)
    )
    summarize(results)
