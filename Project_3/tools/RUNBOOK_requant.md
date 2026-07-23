# RUNBOOK — Requant INT4 đầy đủ

Ký hiệu: **[DOCKER]** = máy nào có Docker cũng chạy được, không cần GPU · **[3060]** = cần GPU của bạn Mr. Senryuu · **[PORTAL]** = nộp lên BTC.

## Số liệu nền (đo từ header `awq_model/model.safetensors`)

| Nhóm | dtype | MB |
|---|---|---:|
| `lm_head` + `embed_tokens` | BF16 ❌ | 536.9 |
| FFN | I32 4-bit ✅ | 418.4 |
| `conv.in_proj` / `out_proj` (10 layer) | BF16 ❌ | 335.7 |
| attention | I32 4-bit ✅ | 32.7 |
| **Tổng file** | | **1323.7** |

**Byte đọc mỗi step decode** = 1323.7 − 268.4 (`embed_tokens` là *gather*, không phải GEMM) = **1055.3 MB** → ở ~600 GB/s = **1.76 ms**.

| Sau bước | File | Đọc/step | Băng thông | Tiết kiệm | Điểm |
|---|---:|---:|---:|---:|---:|
| hiện tại | 1323.7 MB | 1055.3 MB | 1.76 ms | — | — |
| **stage `conv`** | ~1076 MB | **807.6 MB** | 1.35 ms | −0.41 ms | **+3.1** |
| **stage `full`** (thêm lm_head) | ~878 MB | **609.2 MB** | 1.02 ms | −0.33 ms | **+2.4** |

> Mốc kiểm tra kích thước: sau `conv` phải ra **~1.05–1.10 GB**; sau `full` phải ra **~0.85–0.90 GB**. Ra khác nhiều nghĩa là recipe không ăn.

---

# CỬA 0 — [DOCKER] 5 phút. Kiểm tra vLLM có quantize nổi conv proj không

**Đây là cửa quan trọng nhất.** Nếu vLLM khai báo `conv.in_proj/out_proj` bằng `nn.Linear` thuần thì compressed-tensors **sẽ bỏ qua chúng**, và mọi công sức calibration đổ sông.

```bash
# 1. Tìm file model của kiến trúc lfm2
docker run --rm --entrypoint bash vllm/vllm-openai:v0.22.1 -c \
  "ls /usr/local/lib/python3*/dist-packages/vllm/model_executor/models/ | grep -i lfm"

# 2. Xem in_proj/out_proj được khai báo bằng lớp gì
docker run --rm --entrypoint bash vllm/vllm-openai:v0.22.1 -c \
  "grep -nE 'in_proj|out_proj|quant_config|Linear' \
   /usr/local/lib/python3*/dist-packages/vllm/model_executor/models/lfm2.py"
```

**Đọc kết quả:**

| Thấy gì | Kết luận |
|---|---|
| `self.in_proj = MergedColumnParallelLinear(... quant_config=quant_config ...)` | ✅ **Quantize được** → chạy tiếp CỬA 1 |
| `self.in_proj = ReplicatedLinear(... quant_config ...)` | ✅ như trên |
| `self.in_proj = nn.Linear(...)` (không có `quant_config`) | ❌ **KHÔNG quantize được** → sang PHƯƠNG ÁN B ở cuối file |

Làm luôn cùng lúc cho `lm_head` — tìm `ParallelLMHead` và xem có nhận `quant_config` không. Không có thì bỏ stage `full`.

---

# CỬA 1 — [3060] Requant, khoảng 1–2 tiếng

```bash
pip install "llmcompressor==0.13.*" datasets transformers accelerate

cd Project_3
python3 tools/requant_int4_full.py --stage conv --calib-samples 256 --seqlen 2048
```

`--calib-samples 256 --seqlen 2048` là để vừa 12 GB VRAM. GPU lớn hơn thì dùng mặc định (512 / 4096).

Script mặc định dùng **`W4A16` đối xứng**, khác bản cũ (`W4A16_ASYM`). Lý do: checkpoint cũ có `weight_zero_point`, dạng bất đối xứng dễ trượt khỏi kernel **Marlin** nhanh của vLLM — đây có thể mới là nguyên nhân thật khiến "AWQ ≈ FP8".

---

# CỬA 2 — [3060] Kiểm tra kích thước

```bash
du -sh awq_model_int4_conv/
python3 tools/requant_int4_full.py --verify --out awq_model_int4_conv
```

Phải ra **~1.05–1.10 GB**. Nếu vẫn ~1.32 GB ⇒ recipe không ăn vào conv proj ⇒ **dừng, quay lại CỬA 0**.

---

# CỬA 3 — [3060] vLLM nạp được không, và có THẬT SỰ dùng đường quantized không

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v $PWD/awq_model_int4_conv:/model:ro \
  vllm/vllm-openai:v0.22.1 \
  --model=/model --max-model-len=2048 --max-num-seqs=4 2>&1 | tee load.log
```

Kiểm trong `load.log`:

```bash
grep -iE "compressed|marlin|quant|ignore|skip|conv" load.log
```

| Thấy gì | Nghĩa là |
|---|---|
| `Using ... Marlin ... kernel` | ✅ tốt nhất — đường nhanh |
| `CompressedTensorsWNA16` (không Marlin) | ⚠️ chạy được nhưng kernel chậm hơn, lợi ích có thể bị ăn mất |
| conv proj bị liệt kê trong `ignored_layers` | ❌ vẫn không nén — quay lại CỬA 0 |
| lỗi nạp weight | ❌ vLLM chưa hỗ trợ → PHƯƠNG ÁN B |

---

# CỬA 4 — [3060] Đo lại độ chính xác

`f(Δ)=1` khi `accuracy_drop ≤ 0.10`, baseline BF16 = **0.4** ⇒ bản nén phải đạt **≥ 0.30**.

```bash
pip install lm-eval
lm_eval --model vllm \
  --model_args pretrained=./awq_model_int4_conv,max_model_len=4096,gpu_memory_utilization=0.85 \
  --tasks gpqa_diamond_zeroshot --batch_size 4 --seed 1234
```

⚠️ GPQA Diamond chỉ có 198 câu ⇒ sai số chuẩn ~0.035, tức **±0.07 ở mức 2σ**. Một lần đo ra 0.32 **không** đảm bảo an toàn. Chạy 2–3 seed rồi lấy mức thấp nhất mà quyết. Nếu rơi xuống ~0.30 thì quay lại `SCHEME=W4A16_ASYM` (chính xác hơn, đổi lại rủi ro kernel chậm).

---

# CỬA 5 — [3060] Build & push image

```bash
docker build -f Dockerfile.int4 -t <user>/vllm-lfm2-int4:v1 .
docker push <user>/vllm-lfm2-int4:v1
```

`Dockerfile.int4` đang `COPY awq_model_int4_conv/ /model/` — khớp tên thư mục ở CỬA 1.

---

# CỬA 6 — [PORTAL] Nộp và đọc kết quả

Sửa `image:` trong `docker-compose-M1-int4-cudagraph.yml` rồi nộp.

⚠️ File M1 đổi **2 biến** (INT4 + cudagraph FULL). Muốn giữ kỷ luật 1 biến thì nộp **T3 (chỉ cudagraph FULL)** trước, hoặc bỏ dòng `cudagraph_mode` trong M1 để chỉ đo riêng INT4.

Nộp xong:

```bash
python3 tools/infer_tpot.py --score <final_score> --p50 <ttft_p50_ms> --p95 <ttft_p95_ms> --fail <failed_count>
```

**Đối chiếu dự đoán:** stage `conv` phải kéo TPOT hiệu dụng từ **4.125 ms → ~3.71 ms** (điểm ~62.5).
- Đúng như dự đoán ⇒ mô hình băng thông chuẩn, chạy tiếp stage `full`.
- Không nhúc nhích ⇒ **không phải bandwidth-bound**, toàn bộ 4 ms là overhead host ⇒ bỏ nhánh requant, dồn hết vào cudagraph/version.

Đây là phép thử phân định rõ ràng — dù kết quả nào cũng học được một điều chắc chắn.

---

# PHƯƠNG ÁN B — nếu CỬA 0 hoặc CỬA 3 trượt

vLLM không nén được conv proj về INT4 thì hạ mục tiêu xuống **FP8** cho riêng nhóm đó:

- 335.7 MB → **168 MB** (thay vì 88 MB)
- đọc/step 1055 → 887 MB ⇒ **−0.28 ms** ⇒ **+2.1 điểm**

Sửa `build_recipe()` trong `tools/requant_int4_full.py`: đổi `scheme="W4A16"` thành `scheme="FP8"` cho nhóm `conv.*`. Được ít hơn nhưng khả năng vLLM hỗ trợ cao hơn nhiều.

---

# Việc KHÔNG nên làm

**Đừng gỡ `lm_head.weight` trùng lặp** (dù `tie_embedding: true` nên nó trùng với `embed_tokens`). Nó chỉ tiết kiệm **dung lượng đĩa/VRAM**, không giảm byte đọc mỗi step — phép GEMM tính logits vẫn phải đọc đủ ma trận đó. Không được điểm nào. VRAM 18 GB thừa sức chứa, không phải nút thắt.
