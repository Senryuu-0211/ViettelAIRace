# REPORT — A1: W4A16 Symmetric + lm_head INT4

**Date:** 2026-07-24
**Image:** `trisstarn/vllm-lfm2-w4a16:v1`
**Compose:** `docker-compose-A1-w4a16.yml`

---

## Score

| Metric | Baseline (59.32) | A1 | Delta |
|--------|:---:|:---:|:---:|
| `final_score` | 59.32 | **48.90** | −10.42 |
| `ers` | 59.32 | 48.90 | −10.42 |
| `f_delta` | 1.0 | 1.0 | 0 |
| `accuracy_drop` | 0 | 0 | 0 |
| `ttft_p50_ms` | 55 | 55 | 0 |
| `ttft_p95_ms` | 79 | 78 | −1 |
| `tbt_median_ms` | 4 | **6** | +2 |
| `failed_count` | 6 | 7 | +1 |
| `total_count` | 420 | 420 | 0 |

**TPOT hiệu dụng (suy ra): 4.125 ms → ~5.85 ms** — tệ hơn ~1.73 ms.

---

## Analysis

### What went wrong

Thay vì cắt được bandwidth → giảm TPOT như dự báo (+2.4 điểm, TPOT ~3.79 ms), kết quả ngược lại: **TPOT tăng từ 4ms lên 6ms, mất 10.4 điểm.**

`tbt_median_ms = 6` (baseline = 4) cho thấy mỗi token chậm hơn ~2ms. Điều này chỉ có thể giải thích bằng **kernel Marlin W4A16 bị chậm hơn kernel FP8 trên H200**, không phải nhanh hơn như giả định.

### Các giả thuyết

1. **Marlin kernel không tối ưu trên H200 (SM90/Hopper).** Marlin được thiết kế cho Ampere (SM80), có thể không tận dụng được tensor core FP8 của Hopper. Trong khi đó `--quantization=fp8` của baseline dùng kernel native FP8 của Hopper — có thể nhanh hơn đáng kể.

2. **Dequant overhead > bandwidth saving.** Với batch 1-2, mỗi step decode đọc model weight qua kernel Marlin INT4 → giải nén về FP16 → tính GEMM. Chi phí giải nén có thể lớn hơn phần tiết kiệm băng thông, đặc biệt trên H200 vốn đã có băng thông rất cao.

3. **`rms_norm` IR priority.** v0.25.1 log: `IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])`. Không rõ v0.22.1 dùng gì — nếu baseline dùng custom/triton kernel cho norm thì nhanh hơn.

4. **Fail tăng 6→7.** Thêm 1 request lỗi, có thể do checkpoint INT4 gây sai lệch logit ở ngưỡng biên.

---

## 10 Steps Executed

### Step 1 — git pull
```
git pull origin develop
cd Project_3
```
✅ Done. Merge conflict in `docker-compose-A1-w4a16.yml` resolved.

### Step 2 — Environment setup
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --no-deps
.venv\Scripts\python.exe -m pip install llmcompressor==0.11.0 datasets accelerate
```
✅ Done. Installed in isolated venv.

**Errors encountered:**
- `llmcompressor==0.13.*` does not exist. Highest available: 0.12.0. Used 0.11.0 for compatibility.
- `torch` CPU-only initially. CUDA build required manual install from `cu126` channel with `--no-deps` due to `typing-extensions` naming conflict.
- Torch version ping-pong: 2.10.0+cpu → 2.13.0+cu126 → 2.11.0+cpu → 2.10.0+cu126. Final: `torch 2.10.0+cu126`.

### Step 3 — Requant
```
.venv\Scripts\python.exe tools/requant_int4_full.py --stage lmhead --calib-samples 256 --seqlen 2048
```
✅ Done. Output: `awq_model_int4_lmhead/` = 1.126 GB (target: 1.10–1.16 GB).

**Errors encountered:**
- `AWQModifier` default mappings (LLaMA-style) fail on LFM2 architecture. Error: `input_layernorm` not found.
- **Fix:** Added custom LFM2 AWQ mappings:
  - `q_layernorm` → `q_proj, k_proj, v_proj` → FAILED (64-dim vs 2048-dim mismatch)
  - `operator_norm` → `q_proj, k_proj, v_proj` → FAILED (conv layers lack q_proj)
  - **Final fix:** Skip AWQ for attention layers entirely. Use AWQ only for FFN (`ffn_norm → w1,w3`, `w3 → w2`). Attention projections quantized via RTN.
- Added `re:^model\.layers\.\d+\.self_attn\.` to RTN targets → hit `Lfm2RMSNorm` quantization error.
- **Fix:** Use exact Linear names: `q_proj, k_proj, v_proj, out_proj` in RTN targets.
- Unicode error at script exit (cp1252 can't print Vietnamese): cosmetic, checkpoint saved correctly.

### Step 4 — Size check
```
Get-ChildItem -Recurse awq_model_int4_lmhead | Measure-Object -Property Length -Sum
```
✅ Done. 1.049 GB on disk (1.126 GB reported by script). Within target.

### Step 5 — Symmetry check
```python
import json
c = json.load(open("awq_model_int4_lmhead/config.json"))["quantization_config"]
for gname, g in c["config_groups"].items():
    w = g["weights"]
    print(f"{gname}: num_bits={w['num_bits']} symmetric={w.get('symmetric')}")
```
✅ Done. Both groups: `num_bits=4 symmetric=True`.

### Step 6 — vLLM load test
```
docker run --gpus all -v "D:\...\awq_model_int4_lmhead:/model:ro" \
  vllm/vllm-openai:v0.25.1-cu129-ubuntu2404 \
  --model=/model --max-model-len=2048 --max-num-seqs=4 --gpu-memory-utilization=0.75
```
✅ Done. Key log output:
```
Using MarlinLinearKernel for CompressedTensorsWNA16
```
**Errors encountered:**
- OOM on first attempt (GPU 10.98/12.0 GiB free < 0.90 utilization). Fixed with `--gpu-memory-utilization=0.75`.

### Step 7 — GPQA
```
.venv\Scripts\python.exe -m lm_eval --model local-completions \
  --model_args "base_url=http://localhost:8000/v1/completions,model=/model,tokenizer=...awq_model_int4_lmhead,num_concurrent=4" \
  --tasks gpqa_diamond_zeroshot --seed 1234
```
⏸️ NOT RUN. **Error:** `Idavidrein/gpqa` is a gated dataset on HuggingFace. Requires access request at https://huggingface.co/datasets/Idavidrein/gpqa. Token exists but access not yet granted.

### Step 8 — Build & Push
```
docker build -f Dockerfile.int4 -t trisstarn/vllm-lfm2-w4a16:v1 .
docker tag vllm-lfm2-w4a16:v1 trisstarn/vllm-lfm2-w4a16:v1
docker push trisstarn/vllm-lfm2-w4a16:v1
```
✅ Done. Image: 38.4 GB. Build assert: `checkpoint W4A16 doi xung: OK`. Layer check: all 3 symmetry gates passed.

**Changes:** Switched from `FROM vllm/vllm-openai:v0.22.1` to `v0.25.1-cu129-ubuntu2404` (save 15 GB disk, skip pull).

### Step 9 — Nộp Portal
```
# Edited docker-compose-A1-w4a16.yml:
image: trisstarn/vllm-lfm2-w4a16:v1
```
✅ Submitted via Portal.

### Step 10 — Đọc kết quả
```
python tools/infer_tpot.py --score 48.90 --p50 55 --p95 78 --fail 7
```
✅ Done. TPOT hiệu dụng: ~5.85 ms (baseline: 4.125 ms). Regression 1.73 ms.

---

## Conclusion

**A1 failed.** W4A16 symmetric quantization via Marlin kernel is **slower** than FP8 on H200 for this model at batch 1-2. The 2ms penalty outweighs any bandwidth savings.

**Next steps:**
1. Revert to baseline FP8 quantization path (59.32) as fallback.
2. Prioritize T3 (cudagraph FULL) + T4 (vLLM version bump) — these target the host overhead, not the bandwidth.
3. Đường B (conv INT4 + patch vLLM) should be **deferred** until bandwidth-bound is confirmed (P1 probe).
4. Investigate whether `rms_norm` kernel choice in v0.25.1 caused part of the regression (baseline uses v0.22.1).

**Marlin W4A16 hypothesis disproved:** Symmetric quantization alone does NOT yield speedup on Hopper — the FP8 native path may already be optimal.
