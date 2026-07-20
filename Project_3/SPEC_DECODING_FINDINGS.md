# Speculative Decoding trên LFM2.5 hybrid — Chẩn đoán & Kế hoạch test

**Mục tiêu:** hạ **TPOT 4ms → ~2ms**. Với γ=2, mỗi 1ms TPOT ≈ **+8 điểm** → đây là đòn bẩy ROI cao nhất để đẩy 59.33 → ~76-80.

---

## 1. Vì sao bản `docker-compose-fp8-spec.yml` (ngram_gpu) crash/hỏng

Có **3 lỗi chồng nhau**, gốc rễ là một bug thật của vLLM với model **hybrid** (LFM2 = 10 conv + 6 attention layer, có conv-state):

| # | Lỗi | Bằng chứng | Cách sửa |
|---|---|---|---|
| 1 🔴 | **ngram spec làm hỏng conv/SSM-state**: forward chạy trên TẤT CẢ token đề xuất, nhưng khi reject **không rollback state** về vị trí token được chấp nhận → output corrupt/crash | vLLM issue **#39273** (hybrid GDN). Đã fix bởi **PR #40738** ("Fix GDN conv + SSM state corruption with ngram spec decode", +30-40%) | **Nâng vLLM** lên bản chứa #40738 (~05/2026+). 0.22.1 **chưa có** |
| 2 🟠 | `--async-scheduling` + spec **không tương thích** (async overlap step ↔ spec cần verify đồng bộ) | Comment trong `docker-compose.yml`; "ngram_gpu + async compat đang review" | **Bỏ** `--async-scheduling` khi bật spec |
| 3 🟠 | `prompt_lookup_min=2` (default) gây **corruption** trên model Qwen3-class | vLLM issue **#40875** (fix: min=8) | Đặt **`prompt_lookup_min=8`** |
| 4 ⚠️ | **KV-quant × spec = degenerate token loop** | vLLM issue **#40831** | **Không** dùng FP8 KV khi spec |

> **Kết luận:** trên vLLM **0.22.1**, spec decoding hybrid LFM2 gần như chắc chắn cho **output hỏng** (dù bỏ async). Bắt buộc **nâng vLLM**.

---

## 2. Điều kiện tiên quyết: NÂNG vLLM (không phạm luật)

- Luật chỉ yêu cầu **framework = vLLM**; image `0.22.1` chỉ là **baseline**, không bắt buộc. Nâng version vẫn là vLLM → **hợp lệ**.
- Sửa `Dockerfile`:
  ```dockerfile
  FROM vllm/vllm-openai:<tag_mới_đã_PIN>   # >= release chứa PR #40738 (~05/2026+)
  COPY awq_model/ /model/
  ```
- **PIN cứng 1 tag cụ thể** (không `:latest`) — vì (a) reproducibility, (b) luật cấm đổi image sau khi nộp.
- Bản mới còn kèm: "prefix caching cho Mamba hybrid" + "dynamic spec decoding tương thích full CUDA graph" → hợp với config của ta.

---

## 3. Config đề xuất: `docker-compose-spec-v2.yml`

Khác bản cũ ở 4 điểm sửa lỗi trên: nâng vLLM · bỏ async · `prompt_lookup_min=8` · không FP8 KV. Giữ nguyên phần đã chạy tốt (AWQ INT4, CUDA graph FULL, prefix caching, các giới hạn memory).

---

## 4. ⚠️ GATE BẮT BUỘC trước khi nộp — verify output KHÔNG hỏng

Spec trên hybrid từng làm corrupt output → **corrupt = trượt Accuracy Gate = mất trắng f(Δ)**. Trước khi tin config này:

1. **Eyeball**: gửi vài prompt, đọc output có mạch lạc không (không lặp vô hạn, không ký tự rác).
2. **So khớp greedy**: cùng prompt, temperature=0 — output spec phải **giống hệt** bản KHÔNG spec (spec đúng là *lossless*). Nếu khác → state rollback vẫn lỗi → **bỏ**.
3. **GPQA nhanh**: chạy `lm_eval` GPQA (subset) so với baseline, đảm bảo Δ vẫn ~0.
4. Nếu prefix-cache + spec crash (#39809): **bỏ `--enable-prefix-caching`** test lại (mất chút TTFT nhưng vẫn thắng nếu TPOT giảm mạnh).

---

## 5. Kịch bản & quyết định

| Kết quả test H200 | Hành động |
|---|---|
| Spec chạy + output sạch + TPOT giảm | 🎉 Nộp — kỳ vọng ERS ~0.76-0.80 |
| Spec chạy nhưng output khác greedy | Bỏ (corrupt) → giữ submission 59.33 |
| Crash cả khi bỏ prefix-cache | vLLM chưa đủ chín cho LFM2+spec → **giữ 59.33**, chuyển sang tối ưu 0-fail + operating point |

**Fallback an toàn luôn là submission AWQ hiện tại (59.33, f_delta=1).** Không đụng vào nó.

---

## 6. Tuning sau khi spec chạy sạch
- `num_speculative_tokens`: sweep 2→5. Cao hơn = phần thưởng lớn nhưng acceptance thấp trên token mới (câu trả lời sinh ra không nằm trong context) → tối ưu thường 2-4.
- Workload multi-turn (gửi lại history) → n-gram match nhiều ở phần prompt lặp; phần answer mới acceptance thấp → TPOT trung bình kỳ vọng ~2ms, không tới 1ms.

---

## Nguồn
- [vLLM #39273 — Ngram spec corrupts hybrid GDN output](https://github.com/vllm-project/vllm/issues/39273)
- [vLLM #40875 — prompt_lookup_min=2 corruption (fix: 8)](https://github.com/vllm-project/vllm/issues/40875)
- [vLLM #40831 — KV-quant × spec degenerate loops](https://github.com/vllm-project/vllm/issues/40831)
- [vLLM #39809 — Mamba prefix caching + spec crash](https://github.com/vllm-project/vllm/issues/39809)
- [vLLM Blog — Hybrid SSM serving (2026-04)](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg)
