"""
Đo TTFT/TPOT tại chỗ, KHÔNG cần đốt lượt nộp.

VÌ SAO CẦN: Portal chỉ trả `tbt_median_ms` làm tròn thành SỐ NGUYÊN (4ms có thể là 3.5
hoặc 4.49). Ta đang tối ưu một đại lượng gần như không nhìn thấy — đó là lý do 3 lần nộp
vừa rồi không học được gì. Script này cho số tới 0.001 ms.

CHẠY: xem tools/README_do_tai_cho.md

BA THỨ QUAN TRỌNG script này xử lý (nếu tự viết tay dễ sai):
  1. `ignore_eos: true`  -> ép model sinh ĐỦ 200 token. Không có nó, model dừng sớm
     ở ~20 token thì mẫu TPOT quá ít và lệch.
  2. Prefix ngẫu nhiên mỗi request -> KHÔNG cho prefix caching ăn gian TTFT. Chạy lại
     cùng một prompt lần 2 thì TTFT tụt còn vài ms, số đó vô nghĩa.
     Dùng --shared-prefix để đo mặt ngược lại (xem prefix caching giúp được bao nhiêu).
  3. `stream_options.include_usage` -> in ra prompt_tokens THẬT, để xác nhận mình đang
     đo ở ~4000 token đúng như trace, chứ không phải 1500.
"""

import argparse
import json
import random
import statistics
import string
import threading
import time
import urllib.request

PROMPT_TOKENS = 4000
MAX_TOKENS = 200

_LOCK = threading.Lock()


def make_prompt(unique: bool) -> str:
    # ~3 ký tự/token, khớp tỉ lệ in_chars/in_tokens_est = 12000/4000 trong trace thật.
    body = ("Bệnh nhân nam 54 tuổi, tiền sử tăng huyết áp và đái tháo đường type 2, "
            "vào viện vì đau ngực trái lan lên vai. ") * 200
    body = body[: PROMPT_TOKENS * 3]
    if not unique:
        return body
    # Prefix ngẫu nhiên ĐỨNG ĐẦU -> phá vỡ mọi block prefix cache.
    salt = "".join(random.choices(string.ascii_letters, k=64))
    return salt + " " + body


def one_request(base_url, model, unique, out):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": make_prompt(unique)}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,                     # ép sinh đủ 200 token
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    stamps, usage = [], None
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                obj = json.loads(chunk)
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        stamps.append(time.perf_counter())
    except Exception as e:                       # noqa: BLE001
        with _LOCK:
            out.append({"error": repr(e)})
        return

    if len(stamps) < 2:
        with _LOCK:
            out.append({"error": f"chỉ nhận {len(stamps)} token"})
        return

    with _LOCK:
        out.append({
            "ttft_ms": (stamps[0] - t0) * 1000.0,
            "gaps_ms": [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])],
            "usage": usage,
        })


def pct(xs, p):
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="LFM2.5-1.2B-Instruct")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="2 = bằng concurrency trung bình đo từ trace thật")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--label", default="run")
    ap.add_argument("--shared-prefix", action="store_true",
                    help="dùng CHUNG prompt để prefix caching được ăn -> đo mặt ngược lại")
    args = ap.parse_args()

    unique = not args.shared_prefix

    print(f"[{args.label}] warmup 2 request...")
    for _ in range(2):
        one_request(args.base_url, args.model, unique, [])

    results = []
    t_start = time.perf_counter()
    for i in range(args.rounds):
        out, threads = [], []
        for _ in range(args.concurrency):
            t = threading.Thread(target=one_request,
                                 args=(args.base_url, args.model, unique, out))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        results.extend(out)
        print(f"  vòng {i+1}/{args.rounds}", end="\r")

    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    if not ok:
        print("\nKHÔNG có kết quả hợp lệ. Lỗi đầu tiên:", bad[0]["error"] if bad else "?")
        return

    ttfts = sorted(r["ttft_ms"] for r in ok)
    gaps = sorted(g for r in ok for g in r["gaps_ms"])
    u = next((r["usage"] for r in ok if r.get("usage")), None)

    print(f"\n\n{'='*62}")
    print(f"  {args.label}   (concurrency={args.concurrency}, "
          f"{len(ok)} request OK / {len(bad)} lỗi, {len(gaps)} mẫu TPOT, "
          f"{time.perf_counter()-t_start:.0f}s)")
    if u:
        print(f"  prompt_tokens THẬT = {u.get('prompt_tokens')} "
              f"| completion_tokens = {u.get('completion_tokens')}")
        if u.get("prompt_tokens", 0) < 3000:
            print("  ⚠️ prompt ngắn hơn trace thật (~4000) -> TTFT sẽ đẹp giả tạo")
    print(f"{'='*62}")
    print(f"  TTFT   p50 {pct(ttfts,0.50):8.2f} ms   p95 {pct(ttfts,0.95):8.2f} ms")
    print(f"  TPOT   p50 {statistics.median(gaps):8.3f} ms   p95 {pct(gaps,0.95):8.3f} ms"
          f"   mean {statistics.fmean(gaps):8.3f} ms")

    s_ttft = max(0.0, min(1.0, (400 - pct(ttfts, 0.50)) / 390)) ** 2
    s_tpot = max(0.0, min(1.0, (10 - statistics.median(gaps)) / 9)) ** 2
    print(f"\n  s_ttft {s_ttft:.4f}   s_tpot {s_tpot:.4f}")
    print(f"  ==> ƯỚC LƯỢNG ĐIỂM (giả định 0 fail): "
          f"{100*0.5*(s_ttft+s_tpot):.1f}")
    print("  (mốc: bản 59.32 trên Portal có TTFT p50 55ms, TPOT 4ms -> ước lượng 61.3)")
    if bad:
        print(f"\n  {len(bad)} lỗi, ví dụ: {bad[0]['error']}")


if __name__ == "__main__":
    main()
