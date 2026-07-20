"""
Do do dai prefix chung THAT (theo token, khong phai ky tu) giua cac request
trong trace -- de biet tran benefit ly thuyet cua prefix caching.
Chay hoan toan offline, KHONG can server vLLM dang chay, chi can trace + tokenizer.

Cach dung:
    python measure_true_prefix_overlap.py trace-round1.jsonl --tokenizer Qwen/Qwen3.5-2B
"""
import json
import argparse
import statistics


def load_trace(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def get_field(record, candidates, default=None):
    for c in candidates:
        if c in record:
            return record[c]
    return default


def extract_text(record):
    body = record.get("body", {})
    messages = None
    if isinstance(body, dict):
        messages = body.get("messages")
    if not messages:
        messages = get_field(record, ["messages"])
    if messages:
        return "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))
    return get_field(record, ["prompt", "input", "text"], "")


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_path")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    records = load_trace(args.trace_path)
    texts = [extract_text(r) for r in records]
    token_lists = [tok.encode(t) for t in texts]

    # Dung request dau tien lam "mo neo" -- neu prefix la system prompt/template
    # chung, moi request khac se khop voi no o phan dau.
    anchor = token_lists[0]

    shared_lens = []
    ratios = []
    for tl in token_lists:
        shared = common_prefix_len(anchor, tl)
        ratio = shared / len(tl) if tl else 0.0
        shared_lens.append(shared)
        ratios.append(ratio)

    print(f"Tong so request: {len(records)}")
    print(f"\nCommon prefix length (tokens, so voi request #0):")
    print(f"  mean={statistics.mean(shared_lens):.0f}, min={min(shared_lens)}, "
          f"max={max(shared_lens)}, median={statistics.median(shared_lens):.0f}")

    print(f"\nTy le shared/total tokens (tran ly thuyet cua prefix caching):")
    print(f"  mean={statistics.mean(ratios)*100:.1f}%, min={min(ratios)*100:.1f}%, "
          f"max={max(ratios)*100:.1f}%, median={statistics.median(ratios)*100:.1f}%")

    print(f"\n--- Doi chieu voi hit rate do duoc tu vLLM log (~32-33%) ---")
    mean_ratio_pct = statistics.mean(ratios) * 100
    if abs(mean_ratio_pct - 33) < 8:
        print(f"=> Ty le ly thuyet ({mean_ratio_pct:.1f}%) KHOP voi hit rate thuc te (~33%).")
        print("   Ket luan: cache dang hoat dong dung, ~33% la TRAN THAT cua noi dung trung lap,")
        print("   khong phai loi co che. Muon tang ERS phai giam khoi luong prefill THAT,")
        print("   khong the trong cay vao cache them nua.")
    elif mean_ratio_pct > 33 + 8:
        print(f"=> Ty le ly thuyet ({mean_ratio_pct:.1f}%) CAO HON nhieu so voi hit rate thuc te (~33%).")
        print("   Ket luan: cache dang bi mat mat do block-size granularity hoac eviction --")
        print("   dang co du dia de cai thien BANG CACH tune cache (khong phai bang compute).")
    else:
        print(f"=> Ty le ly thuyet ({mean_ratio_pct:.1f}%) THAP HON hit rate thuc te (~33%) --")
        print("   co the do gia dinh 'mo neo' request #0 khong dai dien, can kiem tra lai.")