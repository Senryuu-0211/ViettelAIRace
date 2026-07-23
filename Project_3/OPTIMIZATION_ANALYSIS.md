# Project_3 — Phân tích tối ưu dựa trên KẾT QUẢ THỰC NGHIỆM

## 1. Bảng điểm thật

| Config | Điểm | vLLM | quant | seqs | batched-tok | async | cudagraph | mem | khác |
|---|---|---|---|---|---|---|---|---|---|
| `59.32-fp8` | **59.32** ⬅️ BASELINE | 0.22.1 | fp8 | 16 | 8192 | ✅ | default (piecewise) | 0.90 | — |
| (AWQ trước) | 59.33 | 0.22.1 | AWQ INT4 | 16 | 8192 | ✅ | **FULL** | 0.90 | — |
| `T1-chunk4096` | 58.44 ⬇️ | 0.22.1 | fp8 | 16 | **4096** | ✅ | default | 0.90 | — |
| `56.38` | 56.38 ⬇️ | **0.25.1** | fp8 | **32** | **16384** | ✅ | FULL | **0.95** | **flashinfer** |
| `34.37-spec` | **34.37** ⬇️⬇️ | 0.25.1 | fp8 | 16 | 8192 | **❌** | FULL | 0.90 | **ngram spec** |

**Chi tiết đo:**

| | 59.32 | T1 (4096) |
|---|---|---|
| ERS / final | 59.32 | **58.44** |
| ttft_p50 / p95 | 55 / 79 ms | **58 / 84 ms** |
| tbt_median | **4 ms** | **4 ms** |
| failed_count | 6 / 420 | **6 / 420** |
| f_delta / accuracy_drop | 1 / 0 | 1 / 0 |

---

## 2. Đọc kết quả T1 (chunked prefill) — GIẢ THUYẾT BỊ BÁC BỎ

Hạch toán chênh lệch −0.88đ:
- s_ttft = ((400−x)/390)². Ở 55ms → 0.7825; ở 58ms → 0.7690 → Δ = −0.0135.
- w = 0.5 → ΔS = −0.0068 → **≈ −0.68 điểm** là do TTFT xấu đi.
- Còn lại ≈ −0.2 điểm ⇒ **TPOT không hề cải thiện** (thậm chí nhích xấu/nhiễu).

**Kết luận: giảm `max-num-batched-tokens` = mất TTFT, không được gì về TPOT. Nhánh này ĐÓNG. KHÔNG nộp T2 (2048).**

---

## 3. 🔑 PHÁT HIỆN QUYẾT ĐỊNH: phân tích trace → **concurrency chỉ ~2**

Tính trực tiếp từ `trace_grading_public.jsonl`:

| Đại lượng | Giá trị |
|---|---|
| Tổng request | 420 (70 hội thoại × 6 turn) |
| Input mỗi turn | **~4000 token, KHÔNG tích lũy** (min 3997 / max 4000) |
| Output | max 200 token |
| Trải dài | 70 conv khởi động rải từ t=0 → t=303s, tổng trace ~323s |
| **Concurrency trung bình** | **2.05** |
| **Concurrency đỉnh** | **6** |
| Throughput | ~1.3 req/s |

### Hệ quả (rất lớn)

1. **`--max-num-seqs=16` KHÔNG BAO GIỜ chạm trần.** Đổi 16→8 hay →32 đều **vô nghĩa**. ⇒ **hủy T6.**
2. **Mọi cờ về batching đều vô nghĩa** — GPU rảnh phần lớn thời gian. Bài này là **latency-bound, không phải throughput-bound**.
3. **Prefill hiếm khi đụng nhau** (chỉ ~2 request cùng lúc) ⇒ đúng như T1 đã chứng minh: chunk prefill không cứu TPOT.
4. **Decode gần như luôn chạy ở batch 1–2.**
5. **6 fail KHÔNG do vượt context**: max input = 4000 + 200 out = 4200 < `max-model-len=5120`. Loại bỏ giả thuyết này.

---

## 4. Chẩn đoán TPOT: **host-bound (CPU dispatch), không phải GPU**

Bằng chứng: `tbt_median = 4ms` **BẤT BIẾN** qua mọi cấu hình đã thử —
AWQ INT4 → 4ms · FP8 → 4ms · chunk 4096 → 4ms.

Ở batch 1, một model **1.2B** trên H200 (dù là lát MiG) cần **dưới 1ms** GPU compute mỗi token.
⇒ **~3ms/step là overhead phía host**: launch kernel, vòng lặp scheduler Python, IPC API-server ↔ engine-core, detokenize, đẩy SSE — tất cả bóp trên **3 CPU core**.

> **Đổi hướng chiến lược: ngừng vặn các núm phía GPU (quantization, backend attention, gpu-mem, batching). Chỉ còn một chiến trường: cắt chi phí CPU mỗi step.**

### ⚠️ Đính chính một kết luận cũ
"**CUDA graph FULL trung tính**" là **so sánh nhiễu biến**: nó đặt *AWQ + FULL* (59.33) cạnh *FP8 + mặc định* (59.32) — **đổi 2 biến cùng lúc**, hai hiệu ứng có thể triệt tiêu nhau. Đây đúng là lỗi đã mắc ở bản 56.38. **Kết luận đó không đứng vững và được mở lại (T3).**

### ⚠️ Đính chính con số vá fail
ERS = tổng/420 với 6 giá trị 0 ⇒ trung bình 414 request còn lại = 59.32×420/414 = 60.18.
Vá hết fail → ERS ≈ 60.18 ⇒ **+0.86 điểm**, KHÔNG phải +1.8 như ghi trước đây.

---

## 5. Hàng đợi test mới (1 biến/lần, xếp theo kỳ vọng)

| # | File | Đổi ĐÚNG 1 thứ so với 59.32 | Vì sao |
|---|---|---|---|
| **T3** | `docker-compose-T3-cudagraph-full.yml` | `cudagraph_mode: "FULL"` | Ở batch 1, FULL graph gộp ~16 layer × nhiều kernel thành **1 lần launch** → đánh thẳng vào overhead host. Đòn bẩy cơ học mạnh nhất còn lại. Và nó **chưa từng được test sạch**. |
| **T4** | `docker-compose-T4-vllm0251.yml` | image → `v0.25.1` | Bản mới cắt overhead vòng lặp engine phía CPU — đúng thứ đang thiếu. 56.38 đã làm nhiễu biến này. |
| **T5** | `docker-compose-T5-ompthreads.yml` | `OMP_NUM_THREADS=1` | 3 core; thread pool OpenMP **spin-wait** cướp CPU của scheduler + SSE. Ở batch 1-2, torch CPU song song vô dụng. |

### ❌ Đã loại khỏi hàng đợi
| Bỏ | Lý do |
|---|---|
| T2 (chunk 2048) | T1 đã bác bỏ nhánh chunk; concurrency ~2 nên prefill không đụng nhau |
| max-num-seqs 16→8 | Trần 16 không bao giờ chạm (đỉnh thực tế 6) |
| gpu-mem 0.90→0.95 | KV cache cần rất ít (16 seq × 5120 tok); không phải nút thắt |
| quantization | Đã chứng minh trung tính (AWQ ≈ FP8) |
| flashinfer | Chỉ 6/16 layer là attention, model 1.2B — GPU không phải nút thắt |
| speculative decoding | −25 điểm, đã đo |

**Quy tắc dừng:** nộp T3 trước. Nếu T3 > 59.32 → giữ làm baseline mới rồi mới nộp T4 chồng lên. Nếu T3 < 59.32 → giữ nguyên 59.32 làm gốc, nộp T4 độc lập.

---

## 6. Việc còn lại: 6 fail

- **6/420 cố định qua MỌI config** (59.32 và 58.44 đều đúng 6) ⇒ **nguyên nhân tất định**, không phải timeout ngẫu nhiên hay OOM (những thứ đó sẽ dao động).
- **Không phải vượt context** (đã loại ở §3).
- **Giả thuyết còn lại mạnh nhất:** 6 = **đúng số turn của 1 hội thoại**. Hội thoại `conv_id=0` xuất phát tại **t=0** — ngay lúc server vừa mở cổng, chưa qua request đầu tiên. Nếu turn đầu lỗi/timeout thì cả 6 turn của conv đó = 0 điểm.
- **Cần bạn của Mr. Senryuu lấy log container** (`docker compose logs`) trong lần chấm tới để xác nhận. Giá trị: **+0.86đ**.

---

## 7. 🎯 MỤC TIÊU 80+ — đã có đội đạt **84 điểm**

Trần "65–75" tôi ước tính trước đó **đã bị dữ liệu bác bỏ**. Có đội đạt **84** ⇒ trần thật ≥ 84. Tính ngược:

| Giả định TTFT | s_ttft | s_tpot cần | ⇒ **TPOT cần** |
|---|---|---|---|
| giữ 55 ms | 0.7825 | 0.8975 | **1.47 ms** |
| 30 ms | 0.900 | 0.780 | 2.05 ms |
| 20 ms | 0.949 | 0.731 | 2.31 ms |

**Giá biên tại điểm hiện tại:** TPOT = **7.41 đ/ms** · TTFT = **0.227 đ/ms** (chênh 33 lần).
⇒ Không có đường vòng: **phải kéo TPOT xuống ~1.5–2 ms.**

### 7.1 Hạch toán 4 ms đó đi đâu

MiG `1g.18gb` = 1 memory slice của H200 ⇒ băng thông **~600 GB/s**.

| Thành phần | Ước tính |
|---|---|
| Đọc weight mỗi step (FP8 ~1.4 GB) | **~2.3 ms** |
| Đọc KV cache (6 attn layer × 8 KV head × 4K tok × batch 2) | ~0.16 ms |
| Overhead host: launch kernel, scheduler Python, IPC, SSE | **~1.5 ms** |
| **Tổng** | **≈ 4.0 ms** ✓ khớp số đo |

⇒ Có **HAI** thành phần phải cắt, và chúng độc lập nhau. Trước giờ ta chỉ vặn linh tinh quanh cái thứ ba (batching) vốn không tồn tại.

### 7.2 🔴 PHÁT HIỆN: `awq_model/` mới nén được **1/3** — số đo thật từ header safetensors

| Nhóm | dtype | MB | |
|---|---|---:|---|
| `lm_head` + `embed_tokens` | **BF16** | **536.9** | ❌ chưa nén (2 × 268.4) |
| FFN (w1/w2/w3) | I32 4-bit | 418.4 | ✅ |
| `conv.in_proj` / `conv.out_proj` (10 layer) | **BF16** | **335.7** | ❌ chưa nén |
| attention (q/k/v/out) | I32 4-bit | 32.7 | ✅ |
| **Tổng** | | **1323.7** | |

> **886 MB — 66.9% — vẫn ở dạng float.** Tệ hơn ước tính ban đầu của tôi (604 MB).

Chi tiết đáng chú ý: `lm_head.weight` **và** `model.embed_tokens.weight` **cùng được lưu** dù `tie_embedding: true` — 268.4 MB × 2.

**Byte thật sự phải đọc mỗi step decode** = 1323.7 − 268.4 (`embed_tokens` là *gather*, không phải GEMM) = **~1055 MB** ⇒ ở 600 GB/s = **~1.76 ms**.
Đo được TPOT 4 ms ⇒ **~2.1 ms còn lại là overhead host**.

⚠️ **Đính chính "quantization trung tính":** sai. AWQ ≈ FP8 vì **cả hai đều chưa cắt được bao nhiêu byte**, không phải vì decode không nhạy với băng thông.

⚠️ **Nghi vấn kernel:** checkpoint có `weight_zero_point` (I32) ⇒ **bất đối xứng** (`W4A16_ASYM`). Đường nhanh **Marlin** của vLLM ưu tiên dạng **đối xứng**; asym dễ rơi xuống kernel chậm, ăn hết phần lợi băng thông. ⇒ requant mặc định dùng **`W4A16` đối xứng**.

**Sau khi nén nốt:** conv 335.7 → ~88 MB · lm_head 268.4 → ~70 MB ⇒ đọc **~609 MB/step = ~1.02 ms** (−0.74 ms).

### 7.3 Ngân sách độ chính xác đang bỏ phí 100%

`f(Δ)=1` khi `accuracy_drop ≤ 0.10`. Bản hiện tại đo được **drop = 0** ⇒ ta **không tiêu một chút nào** trong ngân sách. Nén mạnh tay là **miễn phí về điểm** miễn còn dưới ngưỡng. Phải đo lại GPQA Diamond sau requant để xác nhận.

### 7.4 Đường tới 80+

| Đòn bẩy | Cắt được | TPOT |
|---|---|---|
| (hiện tại) | — | **4.00 ms** |
| INT4 đầy đủ: đọc 1055 → 609 MB/step | −0.74 ms | 3.26 ms |
| cudagraph FULL: overhead host 2.1 → ~0.4 ms | −1.70 ms | **~1.58 ms** |

`s_tpot` 0.444 → **0.877** · giữ TTFT 55 ms (`s_ttft` 0.7825) ⇒ `S = 0.830` ⇒ **≈ 83 điểm**, cộng vá 6 fail **+0.86**.

Muốn chạm **84** cần thêm TTFT 55 → ~35 ms. Nhưng prefill 4K token ở FP8 trên lát MiG đã tốn ~46 ms tính toán thuần ⇒ **TTFT chỉ giảm được nếu prefix caching thật sự trúng**. Cần kiểm tra hit rate — nếu prompt các turn dùng cửa sổ trượt thì cache trượt theo và trượt hết.

### 7.5 Trần cập nhật
- **100đ vẫn bất khả thi** (cần MỌI request TTFT ≤ 10 ms, mà riêng prefill đã ~46 ms).
- **Mục tiêu làm việc: 80+.** Có cơ sở vật lý. **84 đã có người đạt ⇒ khả thi.**

---

## 8. Cách làm KHÔNG đốt lượt nộp

`tools/measure_tpot.py` — client streaming, mô phỏng đúng trace (prompt 4000 token, out 200, concurrency 2), in **TPOT tới 0.01 ms** và quy đổi thẳng ra điểm ước lượng.

Portal chỉ trả `tbt_median_ms` **làm tròn số nguyên** (4 ms có thể là 3.5 hoặc 4.49) — ta đang tối ưu thứ mình gần như không nhìn thấy. Chạy script này ngay sau `docker compose up` để **sàng lọc cấu hình trước khi quyết định nộp**:

```
docker compose -f docker-compose-59.32-fp8.yml up -d && python tools/measure_tpot.py --label baseline
docker compose -f docker-compose-T3-cudagraph-full.yml up -d && python tools/measure_tpot.py --label cudagraph-full
```

## 9. Thứ tự hành động

1. **Đo tại chỗ** baseline vs T3 bằng `measure_tpot.py` → biết cudagraph FULL đáng bao nhiêu ms mà không tốn lượt.
2. **Requant** `tools/requant_int4_full.py --stage conv` trên H200 → kiểm tra checkpoint **≤ 0.90 GB**, smoke-test vLLM nạp được.
3. Đo tại chỗ bản INT4 → nếu TPOT ≤ 2.2 ms thì build `Dockerfile.int4` và nộp `docker-compose-M1-int4-cudagraph.yml`.
4. `--stage full` (thêm lm_head) nếu bước 3 chưa đủ — đo lại GPQA vì đây là bước rủi ro accuracy cao nhất.
5. Lấy `docker compose logs` lần chấm để vá 6 fail.
