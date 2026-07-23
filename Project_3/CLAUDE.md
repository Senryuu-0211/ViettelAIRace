# CLAUDE.md — Project 3: LLM Inference Optimization Challenge

Hướng dẫn khi làm việc trong `Project_3/`. Chỉ dẫn ở đây **override** hành vi mặc định.

---

## 0. Giao tiếp (BẮT BUỘC)
1. Gọi người dùng là **"Mr. Senryuu"** ít nhất 1 lần mỗi câu trả lời. Ưu tiên tiếng Việt.
2. Trước khi xin chạy lệnh terminal, ghi 1 dòng phân loại:
   `Lv: SAFE` (chỉ đọc) · `MODERATE` (đổi cục bộ, hoàn tác được) · `CRITICAL` (khó hoàn tác/outward: push, rm -rf, xóa data).

---

## 1. Bài toán

Tối ưu **serving vLLM** cho **LiquidAI/LFM2.5-1.2B-Instruct** trên **1 MiG H200**. Nộp **Docker image** (endpoint OpenAI-compatible) + `docker-compose.yml` qua Portal BTC.

| Hạng mục | Giá trị |
|---|---|
| Hardware chấm | MiG H200: **18GB VRAM, 3 CPU core, 8GB RAM** |
| OS/Driver | Ubuntu 24.04, driver 590.x, CUDA 13.x |
| Model | LFM2.5-1.2B-Instruct — **hybrid**: 16 layer = **10 conv + 6 full_attention**, hidden 2048, 32 head / 8 KV (GQA), vocab 65536 |
| Framework | **CHỈ được dùng vLLM** |
| Baseline image | `vllm/vllm-openai:v0.22.1` (chỉ là baseline, được nâng version) |

**Trace:** 70 hội thoại multi-turn, **330 request chấm + 15 warmup**, arrival Poisson, `think_ms=3000` giữa turn, **~4K token input/turn**, **max 200 token output**. Bản công khai đã lược text (chỉ arrival + token count); BTC giữ prompt thật.

---

## 2. Cách tính điểm (thuộc lòng)

```
Score = 100 × ERS × f(Δ)
ERS   = mean(S_request);  S = 0 nếu lỗi/timeout/0 token
S     = w·s_ttft + (1−w)·s_tpot
s_x   = [clamp((C_x − x)/(C_x − F_x), 0, 1)] ^ γ
```

| Tham số | Giá trị |
|---|---|
| F_ttft / C_ttft | **10ms / 400ms** |
| F_tpot / C_tpot | **1ms / 10ms** |
| γ | **2** (quadratic — gần floor thì dốc) |
| w | **0.5** |
| f(Δ) | Δ≤0.10 → 1.0 · 0.10<Δ<0.16 → tuyến tính · Δ≥0.16 → **0** |
| Accuracy baseline | 0.4 (BF16), đo GPQA Diamond **SAU vòng online** trên ≤5 submission đội chọn |

**Quy đổi thực dụng:** mỗi **1ms TPOT ≈ ±8 điểm**. TTFT hiện dư headroom lớn → **TPOT là chiến trường duy nhất**.

**Anti-cheat:** cấm pre-bake/hardcode, **dual-path**, gaming metrics, gọi mạng ngoài, sửa tokenizer/weights, **tráo image sau khi nộp**.

---

## 3. Trạng thái hiện tại

**Điểm tốt nhất: `59.32` (fp8) ≈ `59.33` (AWQ)** — fallback an toàn, đừng đụng.
- ERS 59.32 · **f_delta = 1** (accuracy_drop = 0) · TTFT **p50 55ms / p95 79ms** · **TPOT median 4ms** ⟵ bottleneck · **6/420 fail**.
- Phân rã: s_ttft ≈ **0.78** (tốt) · s_tpot ≈ **0.44** (yếu).

**Đã đo `T1-chunk4096` → `58.44`** (TTFT p50 58 / p95 84 · TPOT vẫn 4ms · vẫn 6 fail) ⇒ **thua baseline**.

---

## 3b. 🔑 PHÂN TÍCH TRACE — nền tảng cho mọi quyết định sau này

Tính trực tiếp từ `trace_grading_public.jsonl`:

| | |
|---|---|
| 70 hội thoại × 6 turn = **420 request**, trải **323 giây** | ~1.3 req/s |
| Input mỗi turn **~4000 token, KHÔNG tích lũy** (3997–4000) | output max 200 |
| **Concurrency trung bình `2.05`** · **đỉnh `6`** | ⟵ ĐIỀU QUAN TRỌNG NHẤT |

**Hệ quả:** `max-num-seqs=16` **không bao giờ chạm trần**; decode gần như luôn ở **batch 1–2**; GPU rảnh phần lớn thời gian. Bài này **latency-bound, KHÔNG phải throughput-bound** ⇒ **mọi cờ về batching đều vô nghĩa.**

**TPOT là host-bound (CPU), không phải GPU:** `tbt_median = 4ms` **bất biến** qua AWQ / FP8 / chunk4096. Model 1.2B ở batch 1 trên H200 cần **<1ms** GPU compute ⇒ ~3ms còn lại là **overhead phía host**: launch kernel, vòng lặp scheduler Python, IPC API-server↔engine-core, detokenize, đẩy SSE — bóp trên **3 CPU core**.
⇒ **Chiến trường duy nhất: cắt chi phí CPU mỗi step.**

---

## 4. ⛔ ĐÃ CHỨNG MINH — ĐỪNG LÀM LẠI

| Kết luận | Bằng chứng |
|---|---|
| 🔴 **Speculative decoding CÓ HẠI: −25 điểm** | `34.37-spec` vs `59.32`. Token answer sinh mới không nằm trong n-gram context → acceptance thấp → chỉ thêm việc; lại ép bỏ async |
| 🔴 **KHÔNG được bỏ `--async-scheduling`** | Mọi bản ~59 đều có; bản bỏ nó thì sụp |
| 🔴 **Chunked prefill (giảm batched-tokens) VÔ ÍCH** | `T1` 58.44 < 59.32. TTFT xấu đi (−0.68đ hạch toán được), **TPOT không nhúc nhích**. Concurrency ~2 nên prefill hiếm khi đụng nhau ⇒ **KHÔNG nộp T2** |
| ⚪ **Quantization ~vô nghĩa** (AWQ ≈ FP8) | 59.33 ≈ 59.32 → decode **không** bị chặn bởi băng thông weight |
| ⚪ **max-num-seqs / gpu-mem / flashinfer: bỏ qua** | Trần seqs không chạm; KV cache thừa mứa; GPU không phải nút thắt |
| ⚠️ **Gói nhiều thay đổi cùng lúc = mù** | `56.38` đổi 6 thứ → mất 3đ, không biết do đâu |

### ⚠️ Đính chính (đã ghi sai trước đây)
- "**CUDA graph FULL trung tính**" — **SAI PHƯƠNG PHÁP.** Nó so *AWQ+FULL* (59.33) với *FP8+mặc-định* (59.32), tức **đổi 2 biến**; hai hiệu ứng có thể triệt tiêu nhau. **Kết luận này bị rút lại, mở lại thành T3.**
- Vá 6 fail đáng **+0.86đ** (không phải +1.8đ): 59.32×420/414 = 60.18.

> **KỶ LUẬT BẮT BUỘC: mỗi submission chỉ đổi ĐÚNG 1 BIẾN** so với `59.32-fp8`.

---

## 5. Hàng đợi test hiện tại

| # | File | Đổi ĐÚNG 1 thứ | Vì sao |
|---|---|---|---|
| **P1** | `docker-compose-P1-bf16-probe.yml` | **BỎ** `--quantization=fp8` | **CHẨN ĐOÁN, không nhằm ăn điểm** — dự kiến điểm tệ đi. Đọc/step 1472 → 2340 MB. TPOT suy ra **≥5.3 ms ⇒ bandwidth-bound** (requant đáng làm) · **≤4.5 ms ⇒ KHÔNG** (bỏ hẳn requant). Xem §4b |
| **T3** | `docker-compose-T3-cudagraph-full.yml` | `cudagraph_mode: "FULL"` | Ở batch 1, gộp ~16 layer × nhiều kernel thành **1 lần launch** → đánh thẳng overhead host. Chưa từng test sạch |
| **T4** | `docker-compose-T4-vllm0251.yml` | image → `v0.25.1` | Bản mới cắt overhead vòng lặp engine CPU. 56.38 đã làm nhiễu biến này |
| **T5** | `docker-compose-T5-ompthreads.yml` | `OMP_NUM_THREADS=1` | 3 core; OpenMP thread pool **spin-wait** cướp CPU của scheduler + SSE |

**Quy tắc dừng:** nộp T3 trước. T3 > 59.32 → lấy làm baseline mới rồi chồng T4. T3 < 59.32 → quay về gốc 59.32, nộp T4 độc lập.

**6 fail:** **cố định 6/420 qua MỌI config** ⇒ nguyên nhân **tất định**, không phải timeout ngẫu nhiên/OOM. **Không phải vượt context** (4000+200 < 5120). Giả thuyết mạnh nhất: 6 = đúng số turn của **1 hội thoại**, và `conv_id=0` xuất phát tại **t=0** ngay lúc server vừa mở cổng. ⇒ **cần `docker compose logs` của lần chấm** để xác nhận. Giá trị **+0.86đ**.

---

## 6. 🎯 MỤC TIÊU: 80+ (đã có đội đạt **84**)

Trần "65–75" ghi trước đây **đã bị bác bỏ bằng dữ liệu**. Tính ngược từ 84: giữ TTFT 55ms thì cần **TPOT 1.47 ms**; TTFT 30ms thì cần 2.05 ms.
Giá biên: **TPOT 7.41 đ/ms** · TTFT 0.227 đ/ms ⇒ không có đường vòng, phải kéo TPOT xuống ~1.5–2 ms.

### 🔴 `awq_model/` mới nén được 1/3 — ĐO THẬT từ header safetensors (1323.7 MB)

| Nhóm | dtype | MB | |
|---|---|---:|---|
| `lm_head` + `embed_tokens` | **BF16** | **536.9** | ❌ (268.4 × 2, dù `tie_embedding: true`) |
| FFN | I32 4-bit | 418.4 | ✅ |
| `conv.in_proj`/`out_proj` (10 layer) | **BF16** | **335.7** | ❌ |
| attention | I32 4-bit | 32.7 | ✅ |

**886 MB = 66.9% vẫn là float.** Đây là lý do thật của "AWQ ≈ FP8" — cả hai đều chưa cắt được bao nhiêu byte.

⚠️ Có `weight_zero_point` ⇒ **bất đối xứng** ⇒ dễ trượt kernel **Marlin**. Requant nên dùng **`W4A16` đối xứng**.

### Hạch toán 4 ms (MiG 1g.18gb ≈ 600 GB/s)
Đọc mỗi step = 1323.7 − 268.4 (`embed_tokens` là gather) = **1055 MB** ⇒ **1.76 ms** băng thông · KV 0.16 ms · ⇒ **~2.1 ms là overhead host**.

| Đòn bẩy | TPOT |
|---|---|
| hiện tại | 4.00 ms |
| INT4 đầy đủ (đọc 1055 → 609 MB) | 3.26 ms |
| + cudagraph FULL (host 2.1 → ~0.4 ms) | **~1.58 ms** |

`s_tpot` 0.444 → 0.877 ⇒ **≈ 83đ** (giữ TTFT 55ms) + vá fail 0.86.
Script: `tools/requant_int4_full.py`.

### 4a. ✅ MÔ HÌNH ĐÃ HIỆU CHUẨN — P1 đã giải xong câu hỏi băng thông (2026-07-23)

**P1 (bỏ `--quantization=fp8`) → 46.78** · ttft 61/100 · fail 8 · **TPOT suy ra 5.894 ms** (nền 4.125).
+868 MB byte ⇒ **+1.769 ms**. ⇒ **Decode ĐÚNG LÀ bandwidth-bound.**

```
TPOT(ms) = byte_đọc_mỗi_step(MB) / 490.7  +  1.125
                                  ↑ GB/s      ↑ overhead cố định (ms)
```
Khớp cả hai điểm đo với sai số 0.000 ms. Dự đoán tiên nghiệm là 5.57, đo được 5.894 ⇒ mô hình đúng cả chiều lẫn độ lớn.

### 🔑 PHÁT HIỆN LỚN NHẤT: bản AWQ đang mất trắng ~6.5 điểm vì KERNEL

| Cấu hình | đọc/step | TPOT dự báo | điểm dự báo | điểm THẬT |
|---|---:|---:|---:|---:|
| BF16 (P1) | 2340 MB | 5.89 ms | 48.6 | **46.78** ✓ |
| FP8 (59.32) | 1472 MB | 4.12 ms | 59.3 | **59.32** ✓ |
| **AWQ hiện tại (asym)** | **1055 MB** | **3.27 ms** | **65.8** | **59.33** ❌ |

AWQ lẽ ra phải ~65.8. Chênh **6.5 điểm ≈ 0.85 ms** = **chi phí kernel của `W4A16_ASYM`** (có `weight_zero_point` ⇒ không vào được Marlin), ăn đúng bằng phần lợi băng thông.
⇒ **Ưu tiên số 1: requant sang `W4A16` ĐỐI XỨNG.** Không cần vá vLLM, không cần đổi gì khác.

### Lộ trình có số

| Bước | đọc/step | TPOT | điểm (6 fail / 0 fail) |
|---|---:|---:|---:|
| hiện tại (FP8) | 1472 | 4.12 | 59.3 / 60.2 |
| **W4A16 đối xứng** (`--stage lmhead` bỏ lm_head) | 1055 | 3.27 | **65.8 / 66.8** |
| **+ lm_head INT4** (`--stage lmhead`) | 857 | 2.87 | **69.2 / 70.2** |
| **+ conv INT4** (`--stage full`, cần vá) | 609 | 2.37 | **73.8 / 74.8** |

**Trần của riêng nhánh byte ≈ 75.** Muốn 80+ phải cắt thêm **overhead 1.125 ms** (T4 version / T5 OMP / đường ra SSE-IPC). Overhead → 0.5 ms cùng 609 MB ⇒ TPOT 1.74 ⇒ **~80**.

### 6 fail là TRIỆU CHỨNG, không phải bệnh riêng
P1 chậm hơn ⇒ fail **6 → 8**. Tức fail nhạy với độ trễ (timeout), không phải lỗi tất định lúc khởi động như tôi đoán trước đó. ⇒ **TPOT giảm thì fail sẽ tự giảm**, không cần vá riêng.

---

### 4b. 🔴 CHẶN ĐƯỜNG: vLLM KHÔNG quantize được conv proj (đã kiểm chứng 2026-07-23)

`ShortConv.__init__` (`short_conv.py`) **không nhận `quant_config`**; `Lfm2ShortConvDecoderLayer` cũng không truyền vào. Hai Linear đó dùng `UnquantizedLinearMethod`.

| Layer | quant_config? |
|---|---|
| `lm_head` (lfm2.py:473) · attention · FFN | ✅ |
| **`conv.in_proj` / `conv.out_proj`** | ❌ |

- **Hạ conv xuống FP8 cũng KHÔNG cứu được** — vấn đề là layer không có quant method nào, nên nó tìm tensor `weight` còn checkpoint ghi `weight_packed` ⇒ lỗi nạp, bất kể định dạng.
- **Hệ quả:** `--quantization=fp8` đi qua đúng cơ chế đó ⇒ **bản 59.32 (FP8) cũng để conv proj ở BF16.** Cả AWQ lẫn FP8 đều mang nguyên 335.7 MB.

### 🔴 Dữ liệu đang PHẢN BÁC mô hình băng thông

| Cấu hình | FFN | attn | conv | lm_head | Tổng đọc/step | @600 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| BF16 thuần | 1610 | 126 | 336 | 268 | **2340 MB** | 3.90 ms |
| FP8 (59.32) | 805 | 63 | 336 | 268 | **1472 MB** | 2.45 ms |
| AWQ (59.33) | 418 | 33 | 336 | 268 | **1055 MB** | 1.76 ms |

Mô hình dự đoán AWQ nhanh hơn FP8 **0.69 ms ⇒ +5đ**. Thực tế chênh **0.01đ**.
⇒ Hoặc băng thông MiG ≫ 600 GB/s, hoặc kernel `W4A16_ASYM` chậm ăn hết phần lợi. **P1 phân định.**
⇒ **KHÔNG chạy requant trước khi có kết quả P1** (P1 tốn 1 lượt nộp, 0 GB đĩa; requant tốn ~10 GB + 2 tiếng + có thể cả bản vá vLLM).

### Nếu P1 xác nhận bandwidth-bound thì có 2 đường
| Đường | Cắt | Điểm | Rủi ro |
|---|---:|---:|---|
| Quantize `lm_head` (đã có `quant_config` ✅) | −198 MB | +2.4 | thấp, không đụng vLLM |
| Vá `ShortConv` nhận `quant_config` rồi requant conv | −248 MB | +3.1 | ✅ Mr. Senryuu đã quyết **làm, khỏi hỏi BTC**. `tools/patch_shortconv.py` + `Dockerfile.convquant` |

Làm cả hai ⇒ đọc/step 1055 → **609 MB**.

### Ngân sách accuracy đang bỏ phí 100%
`f(Δ)=1` khi drop ≤ 0.10, mà ta đang đo **drop = 0** ⇒ nén mạnh tay là **miễn phí về điểm** miễn còn dưới ngưỡng. Phải đo lại GPQA sau requant.

### Trần
Lên trên 84 cần thêm TTFT 55→35 ms, nhưng prefill 4K trên lát MiG đã tốn ~46 ms tính toán thuần ⇒ chỉ khả thi nếu **prefix caching thật sự trúng** (cần kiểm tra hit rate).
**100đ vẫn bất khả thi** (đòi MỌI request TTFT ≤ 10 ms).

---

## 6b. Đo TẠI CHỖ, đừng đốt lượt nộp
Portal chỉ trả `tbt_median_ms` **làm tròn số nguyên** — ta đang tối ưu thứ gần như không nhìn thấy.
`tools/measure_tpot.py` mô phỏng đúng trace (prompt 4000 tok, out 200, concurrency 2), in TPOT tới **0.01 ms** + quy đổi thẳng ra điểm. Chạy sau `docker compose up` để sàng lọc cấu hình **trước khi** quyết định nộp.

---

## 7. Bản đồ file

| File | Vai trò |
|---|---|
| `docker-compose-59.32-fp8.yml` | **BASELINE TỐT NHẤT** — mọi ablation xuất phát từ đây |
| `docker-compose-56.38.yml` | Bản đổi 6 thứ → tệ hơn (bài học) |
| `docker-compose-34.37-spec.yml` | Bản spec decoding → thảm họa (đừng lặp lại) |
| `docker-compose-T1-chunk4096.yml` | Đã đo **58.44** → thua. Nhánh chunk ĐÓNG |
| `docker-compose-T2-chunk2048.yml` | ⛔ **ĐÃ HỦY — đừng nộp** (T1 bác bỏ giả thuyết) |
| `docker-compose-T3-cudagraph-full.yml` | Ablation tiếp theo #1 — cudagraph FULL trên nền FP8 |
| `docker-compose-T4-vllm0251.yml` | Ablation #2 — chỉ nâng version |
| `docker-compose-T5-ompthreads.yml` | Ablation #3 — `OMP_NUM_THREADS=1` |
| `docker-compose-P1-bf16-probe.yml` | **CHẨN ĐOÁN** — bỏ fp8, phân định bandwidth-bound hay không |
| `docker-compose-M1-int4-cudagraph.yml` | ⏸️ **TẠM HOÃN** — chỉ dùng nếu P1 xác nhận bandwidth-bound |
| `tools/RUNBOOK_requant.md` | Quy trình requant + ngân sách đĩa + vì sao Phương án B hỏng |
| `Dockerfile.int4` | Build image chứa checkpoint INT4 đã requant |
| `tools/requant_int4_full.py` | Nén nốt conv proj (+ tuỳ chọn lm_head). Chạy trên H200 |
| `tools/measure_tpot.py` | Đo TTFT/TPOT tại chỗ tới 0.01ms — sàng lọc không tốn lượt nộp |
| `OPTIMIZATION_ANALYSIS.md` | Bảng thực nghiệm + hàng đợi ablation |
| `SPEC_DECODING_FINDINGS.md` | Hồ sơ vì sao spec hỏng (lịch sử, đã bị dữ liệu bác bỏ) |
| `NEW_PLAN.md` / `SUMMARY.md` | Đề bài gốc Round 2 / tổng quan + Phase 1 |
| `trace_grading_public.jsonl` | Trace công khai (arrival + token count) |
| `trace_tools/simulate_scheduler.py` | Simulator ERS (chạy CPU được, không cần GPU) |
| `awq_model/` | Checkpoint AWQ INT4 (**1.3GB, Git LFS**) |
| `Dockerfile.spec`, `build_and_push_spec.ps1` | Build image vLLM mới (cho nhánh spec — hiện không dùng) |

---

## 8. Ràng buộc môi trường — ⚠️ ĐỌC KỸ, ĐÂY LÀ CHỖ TỪNG GIẢ ĐỊNH SAI

- 🔴 **KHÔNG AI TRONG ĐỘI CÓ H200.** Ghi chú cũ "máy H200 của bạn Mr. Senryuu" là **sai**.
  Thứ duy nhất chạm được MiG H200 là **Portal của BTC**. GPU đội có: **RTX 3060 12GB**.
- ⇒ **Portal LÀ thiết bị đo.** `final_score` có 2 chữ số thập phân và ta biết đúng công thức
  ⇒ dùng **`tools/infer_tpot.py`** đảo ngược ra **TPOT hiệu dụng ~0.03 ms** — mịn hơn nhiều
  so với `tbt_median_ms` làm tròn số nguyên. **Mỗi lần nộp = 1 phép đo tử tế.**
  - Đã suy ra: `59.32` → TPOT **4.125 ms** · `T1-chunk4096` → **4.152 ms** (+0.027, tức T1 KHÔNG cải thiện TPOT).
- **RTX 3060 dùng làm CỔNG KIỂM TRA, không dùng đo hiệu năng:**
  ✅ config có khởi động được không · ✅ checkpoint requant có nạp được không ·
  ✅ cudagraph có capture thật hay âm thầm fallback (đọc log) · ✅ đo lại GPQA.
  ❌ **Đừng tin số TTFT/TPOT đo trên 3060** — khác kiến trúc (SM86 vs SM90), khác băng thông,
  và CPU máy đó mạnh hơn 3 core của MiG nên sẽ **che mất** phần overhead host.
- **Requant chạy được trên 3060**: `--calib-samples 256 --seqlen 2048` (~1-2 tiếng).
- Máy Windows này **không có GPU** → vai trò: **phân tích, soạn config, soạn script**.
- **Docker Desktop thường TẮT** (CLI có, daemon không) → check trước khi build.
- Image vLLM tải ~**8.2GB** (nén). File model dùng **Git LFS** → không dính giới hạn 100MB GitHub.
- Repo: nhánh `develop` là bản chính (có Project_3).

---

## 9. Nguyên tắc làm việc
- **Dựa vào số liệu thật, không đoán.** Mọi khuyến nghị phải truy được về điểm thực nghiệm ở §4.
- **Không hứa 100đ.** Nói trần thật (§6).
- **1 biến/lần.** Không gói nhiều thay đổi.
- Luôn giữ **fallback an toàn** = submission 59.32/59.33 (f_delta=1), không đụng vào.
