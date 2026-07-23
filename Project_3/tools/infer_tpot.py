"""
Suy ngược TPOT THẬT từ kết quả Portal — độ phân giải ~0.05 ms thay vì 1 ms.

VẤN ĐỀ: Portal trả `tbt_median_ms` làm tròn SỐ NGUYÊN. "4 ms" có thể là 3.50 hoặc
4.49 — chênh gần 7 điểm. Nhưng `final_score` lại có 2 chữ số thập phân, và ta biết
chính xác công thức tính điểm. Vậy thì đảo ngược công thức ra.

CÁCH LÀM:
    ERS = (1/N) * sum(S_i),  S_i = 0.5*s_ttft_i + 0.5*s_tpot_i,  fail => S_i = 0
    => mean(S over request KHÔNG fail) = ERS * N / (N - n_fail)
    => mean(s_tpot) = 2*mean(S_ok) - mean(s_ttft)

    mean(s_ttft) không có sẵn, nhưng có p50 và p95 -> khớp một phân phối lognormal
    (median = p50, p95 = p95) rồi lấy kỳ vọng của s_ttft trên phân phối đó.

    Cuối cùng quy về "TPOT hiệu dụng" = giá trị TPOT đồng nhất cho ra đúng mean(s_tpot):
        tpot_eff = 10 - 9*sqrt(mean_s_tpot)

CẢNH BÁO: đây là ƯỚC LƯỢNG. Sai số chính đến từ giả định dạng phân phối TTFT.
Nhưng sai số đó gần như GIỐNG NHAU giữa các lần nộp, nên phần CHÊNH LỆCH giữa hai
submission — thứ ta thực sự cần — đáng tin hơn nhiều so với con số tuyệt đối.

Chạy: python infer_tpot.py            (in bảng các submission đã biết)
      python infer_tpot.py --score 60.1 --p50 55 --p95 79 --fail 6
"""

import argparse
import math

F_TTFT, C_TTFT = 10.0, 400.0
F_TPOT, C_TPOT = 1.0, 10.0
GAMMA, W = 2.0, 0.5


def s_ttft(x):
    return max(0.0, min(1.0, (C_TTFT - x) / (C_TTFT - F_TTFT))) ** GAMMA


def s_tpot(y):
    return max(0.0, min(1.0, (C_TPOT - y) / (C_TPOT - F_TPOT))) ** GAMMA


def mean_s_ttft(p50, p95, n=20001):
    """Kỳ vọng s_ttft dưới lognormal khớp (median=p50, p95=p95)."""
    if p95 <= p50:
        return s_ttft(p50)
    mu = math.log(p50)
    sigma = math.log(p95 / p50) / 1.6448536269514722  # z_{0.95}
    # Tích phân theo phân vị: E[f(X)] = trung binh f(quantile(u)) voi u ~ U(0,1)
    tot = 0.0
    for i in range(1, n + 1):
        u = (i - 0.5) / n
        # inverse-CDF chuan tac (Acklam approx du dung o day)
        z = _norm_ppf(u)
        tot += s_ttft(math.exp(mu + sigma * z))
    return tot / n


def _norm_ppf(u):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if u < pl:
        q = math.sqrt(-2 * math.log(u))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if u > ph:
        q = math.sqrt(-2 * math.log(1 - u))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = u - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def infer(score, p50, p95, fail, total=420):
    ers = score / 100.0
    mean_S_ok = ers * total / (total - fail)
    m_ttft = mean_s_ttft(p50, p95)
    m_tpot = (mean_S_ok - W * m_ttft) / (1.0 - W)
    m_tpot = max(0.0, min(1.0, m_tpot))
    tpot_eff = C_TPOT - (C_TPOT - F_TPOT) * math.sqrt(m_tpot)
    return m_ttft, m_tpot, tpot_eff


def gain_if(score, p50, p95, fail, new_tpot, total=420, fix_fails=False):
    """Điểm dự kiến nếu TPOT hiệu dụng xuống new_tpot (và tuỳ chọn vá hết fail)."""
    m_ttft, _, _ = infer(score, p50, p95, fail, total)
    S = W * m_ttft + (1 - W) * s_tpot(new_tpot)
    n_fail = 0 if fix_fails else fail
    return 100.0 * S * (total - n_fail) / total


KNOWN = [
    ("59.32-fp8  (BASELINE)", 59.32, 55, 79, 6),
    ("T1-chunk4096",          58.44, 58, 84, 6),
    ("P1-bf16-probe",         46.78, 61, 100, 8),
]

# ── MÔ HÌNH ĐÃ HIỆU CHUẨN (khớp 2 điểm đo thật, 2026-07-23) ──────────────────
#   TPOT(ms) = byte_đọc_mỗi_step(MB) / BANDWIDTH + OVERHEAD
#   Khớp từ: FP8 1472 MB -> 4.125 ms  và  BF16 2340 MB -> 5.894 ms
#   Sai số tái tạo: 0.000 ms ở cả hai điểm.
BANDWIDTH = 490.7   # MB/ms = GB/s — băng thông hiệu dụng của lát MiG
OVERHEAD = 1.125    # ms — chi phí cố định mỗi step, KHÔNG phụ thuộc số byte
                    # (dispatch, scheduler, IPC, detokenize, đẩy SSE trên 3 core)

# Byte phải ĐỌC mỗi step decode cho từng cấu hình.
# Không tính embed_tokens (268.4 MB): nó là phép gather, không phải GEMM.
READ_MB = {
    "BF16 thuần":                       2340,
    "FP8 (bản 59.32)":                  1472,
    "AWQ hiện tại (asym)":              1055,
    "AWQ đối xứng + Marlin":            1055,
    "  + lm_head INT4":                  857,
    "  + conv INT4 (cần vá vLLM)":       609,
}


def tpot_from_bytes(read_mb):
    return read_mb / BANDWIDTH + OVERHEAD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=float)
    ap.add_argument("--p50", type=float)
    ap.add_argument("--p95", type=float)
    ap.add_argument("--fail", type=int, default=0)
    ap.add_argument("--total", type=int, default=420)
    args = ap.parse_args()

    rows = KNOWN if args.score is None else \
        [("(nhập tay)", args.score, args.p50, args.p95, args.fail)]

    print(f"{'Submission':24} {'score':>7} {'ttft p50/p95':>13} {'fail':>5} "
          f"{'mean s_ttft':>12} {'mean s_tpot':>12} {'TPOT hiệu dụng':>16}")
    print("-" * 96)
    base = None
    for name, sc, p50, p95, fail in rows:
        mt, mp, te = infer(sc, p50, p95, fail, args.total)
        delta = "" if base is None else f"   ({te-base:+.3f} ms so với baseline)"
        if base is None:
            base = te
        print(f"{name:24} {sc:7.2f} {p50:6.0f}/{p95:<6.0f} {fail:5d} "
              f"{mt:12.4f} {mp:12.4f} {te:13.3f} ms{delta}")

    if args.score is None:
        print("\n--- Điểm dự kiến nếu kéo được TPOT xuống (giữ nguyên TTFT) ---")
        name, sc, p50, p95, fail = KNOWN[0]
        print(f"{'TPOT hiệu dụng':>16} {'còn 6 fail':>13} {'vá hết fail':>13}")
        for y in (3.5, 3.0, 2.5, 2.0, 1.8, 1.5, 1.2):
            a = gain_if(sc, p50, p95, fail, y)
            b = gain_if(sc, p50, p95, fail, y, fix_fails=True)
            print(f"{y:13.1f} ms {a:13.1f} {b:13.1f}")
        print(f"\n--- Dự báo theo MÔ HÌNH ĐÃ HIỆU CHUẨN "
              f"(BW {BANDWIDTH:.0f} GB/s, overhead {OVERHEAD:.3f} ms) ---")
        print(f"{'Cấu hình':30}{'đọc MB':>8}{'TPOT':>10}{'điểm 6 fail':>13}{'0 fail':>9}")
        for cfg, mb in READ_MB.items():
            y = tpot_from_bytes(mb)
            print(f"{cfg:30}{mb:8.0f}{y:8.2f}ms"
                  f"{gain_if(sc, p50, p95, fail, y):13.1f}"
                  f"{gain_if(sc, p50, p95, fail, y, fix_fails=True):9.1f}")
        print("  ⚠️ 'AWQ hiện tại (asym)' ĐO ĐƯỢC 59.33 chứ không phải ~66 như dự báo")
        print("     -> chênh ~6.5đ (~0.85 ms) là CHI PHÍ KERNEL của W4A16_ASYM.")
        print("     -> requant sang W4A16 ĐỐI XỨNG để vào Marlin là ưu tiên số 1.")

        print("\n--- Cần TTFT bao nhiêu để chạm 84 (ứng với từng mức TPOT) ---")
        mt, _, _ = infer(sc, p50, p95, fail)
        for y in (2.0, 1.8, 1.5, 1.2):
            need_S = 0.84
            need_s_ttft = (need_S - (1 - W) * s_tpot(y)) / W
            if need_s_ttft > 1:
                print(f"  TPOT {y:.1f} ms -> KHÔNG THỂ dù TTFT = 0")
                continue
            x = C_TTFT - (C_TTFT - F_TTFT) * math.sqrt(max(0.0, need_s_ttft))
            print(f"  TPOT {y:.1f} ms -> cần TTFT ~{x:5.1f} ms "
                  f"(hiện p50 = {p50:.0f} ms)")


if __name__ == "__main__":
    main()
