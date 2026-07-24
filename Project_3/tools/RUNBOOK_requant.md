# RUNBOOK — nén model để hạ TPOT

Ký hiệu: **[DOCKER]** máy nào có Docker cũng được · **[GPU]** cần RTX 3060 · **[PORTAL]** nộp lên BTC.

---

## ⭐ TRẠNG THÁI HIỆN TẠI (2026-07-24) — ĐỌC TRƯỚC

**Nền tốt nhất giờ là `T4` = 60.17** (FP8 + vLLM **v0.25.1**, TTFT 52/71, fail 5). Mọi thứ so từ đây, **không phải 59.32**.

### Những gì đã chốt bằng số thật
| Việc | Kết quả | Nghĩa |
|---|---|---|
| P1 (bỏ fp8 → BF16) | 46.78, TPOT 5.89ms | decode **bandwidth-bound** (byte có ảnh hưởng) |
| T4 (fp8, v0.25.1) | **60.17** ⬆️ | **v0.25.1 tốt hơn v0.22.1** — dùng nó làm nền |
| A1 (W4A16, lỡ chạy v0.25.1) | 48.90 ⬇️ | **Marlin INT4 chậm hơn FP8 trên Hopper** — phí dequant ăn hết lợi băng thông |

### 🔴 Marlin W4A16 = ĐÓNG. Nhưng INT4 CHƯA đóng.
Có đội đạt **84** ⇒ cần TPOT ~1.54ms ⇒ đọc/step ~510MB ⇒ **bắt buộc INT4 sâu**. FP8 vật lý không xuống nổi 510MB. ⇒ **INT4 nhanh chắc chắn chạy được trên MiG này**, chỉ là vLLM **chọn nhầm kernel** (Marlin thay vì **Machete**, kernel INT4 native cho Hopper SM90).

### 👉 VIỆC CẦN LÀM, THEO THỨ TỰ

**① [GPU] Chẩn đoán vì sao vLLM không chọn Machete — KHÔNG tốn lượt nộp.** Việc quan trọng nhất; nó quyết định 80+ còn khả thi hay ta chốt ~66.
```bash
docker run --rm --gpus all -v "$PWD/awq_model_int4_lmhead:/model:ro" \
  vllm/vllm-openai:v0.25.1-cu129-ubuntu2404 \
  --model=/model --max-model-len=2048 --max-num-seqs=4 --gpu-memory-utilization=0.75 \
  2>&1 | grep -iE "machete|marlin|can_implement|not supported|falling back|scalar_type|act_order|capability|sm_|group"
```
Gửi lại toàn bộ dòng khớp. Nghi ngờ (theo khả năng): (a) MiG báo compute capability lạ nên Machete từ chối; (b) layout checkpoint của `llmcompressor` không hợp Machete → phải requant với target khác; (c) group_size/act_order kén.

**② [PORTAL] Nộp T5 để cắt overhead host — song song, độc lập với Machete.**
Trước khi nộp, **đổi base image T5 sang v0.25.1** để nó chỉ khác T4 đúng 1 biến (`OMP_NUM_THREADS=1`). Nhắm vào phần overhead 1.125ms (nửa TPOT mà nén byte không đụng tới). Kỳ vọng khiêm tốn +1~2đ.

**③ Giữ T4 làm fallback nộp cuối** — đã hơn 59.32 mọi mặt, `f_delta=1`.

### Trần thật (nền T4)
- Machete hỏng, chỉ cắt được overhead → **~63–66**.
- Machete chạy + nén conv (Đường B) → ~75.
- Machete + conv + overhead xuống ~0.6ms → **~80**.
- **Machete là con đường DUY NHẤT tới 80+.**

### ⛔ Đừng lặp lại
- Đừng nộp checkpoint Marlin W4A16 nữa (A1 đã đo −11đ).
- Đừng đổi 2 biến/lần: build image phải **`FROM v0.25.1`** để khớp nền T4 (A1 hỏng vì lỡ đổi cả version).
- Phần "Đường A/B" bên dưới chỉ có giá trị **SAU KHI** bước ① xác nhận Machete chạy được. Nếu Machete hỏng, toàn bộ nhánh nén byte đóng.

---

<details>
<summary>📦 Chi tiết requant (Đường A / B) — chỉ dùng khi Machete đã chạy</summary>

---

## 0. Bối cảnh trong 10 dòng

- Điểm tốt nhất **59.32**. Nút thắt là **TPOT = 4.125 ms** (suy ra từ điểm, không phải số làm tròn của Portal).
- Đổi 1 ms TPOT ≈ **7.4 điểm**. Đổi 1 ms TTFT chỉ ≈ 0.23 điểm ⇒ **chỉ đánh vào TPOT**.
- Mục tiêu **80+**: cần TPOT xuống **~1.8 ms** (có đội đạt 84 rồi, nên khả thi).
- Concurrency thật của trace chỉ **~2** ⇒ decode chạy ở batch 1-2 ⇒ mọi cờ về batching vô nghĩa (đã chứng minh: chunked prefill thua).
- 4.125 ms gồm 2 phần: **đọc weight** (byte) + **overhead host** (CPU 3 core). Runbook này lo phần byte. Phần overhead lo bằng `docker-compose-T3-cudagraph-full.yml`.
- Checkpoint `awq_model/` hiện tại **mới nén được 1/3**: 886/1324 MB vẫn là float.

---

## 1. ✅ P1 ĐÃ CHẠY XONG — kết quả: LÀM ĐI

**P1 → 46.78** (ttft 61/100, fail 8) ⇒ **TPOT 5.894 ms**, so với nền 4.125 ms.
+868 MB byte cho **+1.769 ms** ⇒ **decode ĐÚNG LÀ bandwidth-bound.**

Mô hình đã hiệu chuẩn, khớp cả hai điểm đo với sai số 0.000 ms:

```
TPOT(ms) = byte_đọc_mỗi_step(MB) / 490.7 GB/s  +  1.125 ms
```

### 🔑 Và nó phát hiện ra ~6.5 điểm đang bị mất trắng

| Cấu hình | đọc/step | TPOT dự báo | điểm dự báo | điểm THẬT |
|---|---:|---:|---:|---:|
| BF16 (P1) | 2340 MB | 5.89 ms | 48.6 | **46.78** ✓ |
| FP8 (59.32) | 1472 MB | 4.12 ms | 59.3 | **59.32** ✓ |
| **AWQ hiện tại** | **1055 MB** | **3.27 ms** | **65.8** | **59.33** ❌ |

Bản AWQ lẽ ra phải ~65.8 mà chỉ được 59.33. Chênh **6.5 điểm ≈ 0.85 ms** chính là **chi phí kernel của `W4A16_ASYM`** — checkpoint có `weight_zero_point` nên không vào được đường nhanh Marlin, và phần chậm đó ăn đúng bằng phần lợi băng thông.

**⇒ Việc đáng làm nhất: requant sang `W4A16` ĐỐI XỨNG.** Script đã mặc định `SCHEME=W4A16`.

### Lộ trình có số

| Bước | đọc/step | TPOT | điểm (6 fail / 0 fail) |
|---|---:|---:|---:|
| hiện tại (FP8) | 1472 | 4.12 ms | 59.3 / 60.2 |
| **W4A16 đối xứng** | 1055 | 3.27 ms | **65.8 / 66.8** |
| **+ lm_head** (Đường A) | 857 | 2.87 ms | **69.2 / 70.2** |
| **+ conv** (Đường B, cần vá) | 609 | 2.37 ms | **73.8 / 74.8** |

Trần của riêng nhánh nén byte ≈ **75**. Muốn 80+ phải cắt thêm **overhead 1.125 ms** bằng T4/T5.

### Tin phụ: 6 fail sẽ tự giảm
P1 chậm hơn ⇒ fail **6 → 8**. Fail nhạy với độ trễ (timeout), không phải lỗi tất định. TPOT giảm thì fail tự giảm — không cần vá riêng.

---

<details>
<summary>Lý do ban đầu phải chạy P1 (giữ lại làm tham chiếu)</summary>

## 1-cũ. ⛔ TRƯỚC KHI TỐN CÔNG: chạy P1

**Có một mâu thuẫn chưa giải.** Byte đọc mỗi step:

| Cấu hình | FFN | attn | conv | lm_head | **Tổng đọc/step** | @600 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| BF16 thuần | 1610 | 126 | 336 | 268 | **2340 MB** | 3.90 ms |
| FP8 (bản 59.32) | 805 | 63 | 336 | 268 | **1472 MB** | 2.45 ms |
| AWQ (bản 59.33) | 418 | 33 | 336 | 268 | **1055 MB** | 1.76 ms |

Mô hình băng thông dự đoán AWQ nhanh hơn FP8 **0.69 ms ⇒ +5 điểm**. Thực tế **59.33 vs 59.32 chênh 0.01**.
⇒ Hoặc băng thông MiG cao hơn 600 GB/s nhiều, hoặc kernel `W4A16_ASYM` chậm ăn hết phần lợi.

**Toàn bộ runbook này chỉ có giá trị nếu decode thật sự bị chặn bởi băng thông.** P1 trả lời câu đó, tốn **1 lượt nộp và 0 GB đĩa**.

### [PORTAL] Nộp `docker-compose-P1-bf16-probe.yml`
Nó chỉ bỏ đúng một dòng `--quantization=fp8`. **Điểm sẽ tệ đi — đó là chuyện bình thường**, đây là phép đo chứ không phải bản ăn điểm.

### Đọc kết quả
```bash
python3 tools/infer_tpot.py --score <final_score> --p50 <ttft_p50_ms> --p95 <ttft_p95_ms> --fail <failed_count>
```

| TPOT suy ra | Kết luận | Làm gì |
|---|---|---|
| **≥ 5.3 ms** | Bandwidth-bound ✅ | Làm tiếp mục 2 trở đi |
| **≤ 4.5 ms** | KHÔNG bandwidth-bound ❌ | **DỪNG runbook này.** Cả 4.125 ms là overhead host → dồn hết vào T3 / T4 / T5 |

> Song song, nộp luôn `docker-compose-T3-cudagraph-full.yml` (chỉ thêm `cudagraph_mode: "FULL"`). Cũng miễn phí, và tấn công nửa còn lại.

</details>

---

## 2. Hai đường đi, chọn theo mức chấp nhận rủi ro

| | Đường A — chỉ `lm_head` | Đường B — thêm cả conv proj |
|---|---|---|
| Vá vLLM? | **Không** | **Có** (`tools/patch_shortconv.py`) |
| Cắt được | −198 MB | −446 MB |
| Đọc/step | 1055 → **857 MB** | 1055 → **609 MB** |
| TPOT dự kiến | 1.76 → **1.43 ms** | 1.76 → **1.02 ms** |
| Điểm dự kiến | **+2.4** | **+5.5** |
| File ra | ~1.13 GB | ~0.88 GB |
| Lệnh | `--stage lmhead` | `--stage full` |

**Vì sao conv proj cần vá:** `ShortConv.__init__` trong vLLM không nhận `quant_config`, nên `in_proj`/`out_proj` dùng `UnquantizedLinearMethod` và đi tìm tensor tên `weight`. Checkpoint nén ghi ra `weight_packed`/`weight_scale` ⇒ **nạp lỗi, bất kể nén INT4 hay FP8**. Đây cũng là lý do bản FP8 59.32 đang để nguyên 335.7 MB conv ở BF16.

**Khuyến nghị:** làm **A trước** (không rủi ro, xác nhận cả quy trình chạy thông), rồi mới B.

---

## 3. Ngân sách ổ đĩa

| Hạng mục | GB |
|---|---:|
| Image `vllm/vllm-openai:v0.22.1` | ~15 (thường đã có) |
| Image `v0.25.1-cu129` (nhánh spec cũ) | ~15 — **xoá được** |
| Model BF16 gốc | 2.4 |
| pip torch CUDA + llmcompressor | ~7 mới / ~1 nếu đã có torch |
| Dataset calibration | 0.05 (nhờ streaming, mặc định) |
| Checkpoint xuất ra | ~1.1 mỗi stage |
| Image mới build | ~1.1 |
| Cache build | ~2 |

**Thực tế: ~8–10 GB** nếu đã có image + torch. Máy trắng: ~30 GB.

Lấy lại chỗ, theo thứ tự hiệu quả:
1. **Docker Desktop → Settings → Resources → Advanced → Disk image location → ổ `D:`** (mạnh nhất, kéo cả chục GB khỏi C: vĩnh viễn — và né luôn chuyện `.vhdx` của WSL2 không tự co lại)
2. `docker system df` rồi `docker image rm vllm/vllm-openai:v0.25.1-cu129-ubuntu2404` (~15 GB) + `docker builder prune -a`
3. `export HF_HOME=/d/hf_cache` — model + dataset không đụng ổ C:
4. Đo GPQA qua endpoint container, **đừng cài `vllm` vào python** (tiết kiệm ~4 GB) — xem CỬA 4

---

## 4. ĐƯỜNG A — chỉ nén `lm_head`

### A1 [GPU] Requant (~1–2 tiếng)
```bash
pip install "llmcompressor==0.13.*" datasets transformers accelerate
cd Project_3
python3 tools/requant_int4_full.py --stage lmhead --calib-samples 256 --seqlen 2048
```
`--calib-samples 256 --seqlen 2048` để vừa 12 GB VRAM. Dataset dùng streaming nên chỉ tải ~50 MB.

Recipe mặc định **`W4A16` đối xứng**, khác checkpoint cũ (`W4A16_ASYM`). Lý do: dạng bất đối xứng có `weight_zero_point`, dễ trượt khỏi kernel **Marlin** — nghi là nguyên nhân thật khiến "AWQ ≈ FP8".

### A2 [GPU] Kiểm kích thước
```bash
du -sh awq_model_int4_lmhead/
```
Phải ra **~1.13 GB** (từ 1.32 GB). Vẫn 1.32 GB ⇒ recipe không ăn vào `lm_head`, dừng lại xem `config.json` mục `ignore`.

### A3 [GPU] Nạp thử + xem đúng kernel chưa
```bash
docker run --rm --gpus all -v "$PWD/awq_model_int4_lmhead:/model:ro" \
  vllm/vllm-openai:v0.22.1 --model=/model --max-model-len=2048 --max-num-seqs=4 \
  2>&1 | tee load.log

grep -iE "marlin|compressed|quant|ignore|lm_head" load.log
```

| Thấy gì | Nghĩa |
|---|---|
| `Marlin` kernel | ✅ tốt nhất |
| `CompressedTensorsWNA16` không kèm Marlin | ⚠️ chạy được nhưng chậm hơn — lợi ích có thể bị ăn mất |
| `lm_head` nằm trong `ignored_layers` | ❌ chưa nén, quay lại A1 |
| lỗi nạp weight | ❌ vLLM không hỗ trợ nén `lm_head` → bỏ đường A, sang B |

### A4 [GPU] Đo lại độ chính xác
`f(Δ)=1` khi drop ≤ 0.10, baseline BF16 = 0.4 ⇒ **phải đạt ≥ 0.30**.

Giữ container A3 đang chạy, rồi:
```bash
pip install lm-eval
lm_eval --model local-completions \
  --model_args base_url=http://localhost:8000/v1/completions,model=/model,num_concurrent=4 \
  --tasks gpqa_diamond_zeroshot --seed 1234
```

⚠️ GPQA Diamond chỉ **198 câu** ⇒ sai số chuẩn ~0.035, tức **±0.07 ở 2σ**. Một lần đo ra 0.32 **không** đảm bảo an toàn. **Chạy 2–3 seed, lấy mức thấp nhất mà quyết.** Rơi xuống ~0.30 thì đổi `SCHEME=W4A16_ASYM` (chính xác hơn, đổi lại rủi ro kernel chậm).

### A5 [GPU] Build & push
```bash
docker build -f Dockerfile.int4 -t <user>/vllm-lfm2-int4:v1 .
docker push <user>/vllm-lfm2-int4:v1
rm -rf awq_model_int4_lmhead    # nội dung đã nằm trong image
```

### A6 [PORTAL] Nộp & đọc
Sửa `image:` trong `docker-compose-M1-int4-cudagraph.yml`.

⚠️ File M1 đang đổi **2 biến** (checkpoint mới + `cudagraph_mode: FULL`). Giữ kỷ luật 1 biến thì **bỏ dòng `cudagraph_mode`** đi, để đo riêng phần nén.

```bash
python3 tools/infer_tpot.py --score ... --p50 ... --p95 ... --fail ...
```

**Đối chiếu:** TPOT hiệu dụng phải đi từ **4.125 → ~3.79 ms** (điểm ~61.8).
- Đúng ⇒ mô hình chuẩn, sang đường B.
- Không nhúc nhích ⇒ mâu thuẫn ở mục 1 nghiêng về "kernel chậm" hoặc "không bandwidth-bound" ⇒ **dừng, đừng làm B**.

---

## 5. ĐƯỜNG B — vá vLLM rồi nén thêm conv proj

### B1 [DOCKER] Xem trước bản vá đổi gì (không build, không ghi)
```bash
docker run --rm -v "$PWD/tools:/t" --entrypoint python3 vllm/vllm-openai:v0.22.1 \
  /t/patch_shortconv.py --vllm-dir /usr/local/lib/python3.12/dist-packages/vllm --check
```

Phải thấy đúng 4 chỗ thêm vào, **không có gì khác**:
1. `ShortConv.__init__` thêm tham số `quant_config=None`
2. `self.in_proj = ...` thêm `quant_config=quant_config`
3. `self.out_proj = ...` thêm `quant_config=quant_config`
4. `lfm2.py`: `ShortConv(...)` thêm `quant_config=quant_config`

Script **assert đúng 1 lần khớp** cho mỗi mỏ neo. Nguồn vLLM khác giả định ⇒ nó **báo lỗi**, không sửa bừa.

### B2 [GPU] Requant đầy đủ
```bash
python3 tools/requant_int4_full.py --stage full --calib-samples 256 --seqlen 2048
du -sh awq_model_int4_full/          # mốc: ~0.88 GB
```

### B3 [GPU] Build image đã vá
```bash
docker build -f Dockerfile.convquant -t <user>/vllm-lfm2-convquant:v1 .
```
Dockerfile có sẵn bước `import vllm.model_executor.models.lfm2` ⇒ vá sai cú pháp thì **chết lúc build**, không chết lúc chấm.

### B4 [GPU] Nạp thử — đây là chỗ dễ vỡ nhất
```bash
docker run --rm --gpus all <user>/vllm-lfm2-convquant:v1 \
  --model=/model --max-model-len=2048 --max-num-seqs=4 2>&1 | tee load_b.log
grep -iE "conv|in_proj|missing|unexpected|KeyError|marlin" load_b.log
```

**Rủi ro đã biết:** `in_proj` là `MergedColumnParallelLinear` gộp 3 đầu ra (B, C, x của conv). Khi nén, tên tensor đổi `weight` → `weight_packed`/`weight_scale`, mà `load_weights` trong `lfm2.py` có bảng ánh xạ tên riêng (`stacked_params_mapping`). **Script vá KHÔNG xử lý chỗ này.** Thấy `KeyError` hoặc `missing weight` cho `in_proj` thì đó là nguyên nhân — phải sửa thêm bảng ánh xạ đó bằng tay.

### B5 Lặp lại A4 (GPQA) → A5 (build/push) → A6 (nộp + `infer_tpot`)
**Đối chiếu:** TPOT phải đi từ 4.125 → **~3.38 ms** (điểm ~65.5).

---

## 6. Bảng theo dõi — điền vào sau mỗi lần nộp

| Bản | Điểm | ttft p50/p95 | fail | **TPOT suy ra** | Ghi chú |
|---|---|---|---|---|---|
| `59.32-fp8` | 59.32 | 55/79 | 6 | **4.125 ms** | nền, fallback an toàn |
| `T1-chunk4096` | 58.44 | 58/84 | 6 | **4.152 ms** | thua — nhánh chunk đóng |
| `P1-bf16-probe` | | | | | ≥5.3 ⇒ bandwidth-bound |
| `T3-cudagraph-full` | | | | | |
| Đường A (`lmhead`) | | | | | kỳ vọng ~3.79 ms |
| Đường B (`full`) | | | | | kỳ vọng ~3.38 ms |

---

## 7. Đừng làm

- **Đừng requant trước khi có kết quả P1.** P1 tốn 1 lượt nộp và 0 GB; requant tốn ~10 GB + 2 tiếng.
- **Đừng gỡ `lm_head.weight` trùng lặp với `embed_tokens`** (dù `tie_embedding: true`). Chỉ tiết kiệm đĩa/VRAM, **không** giảm byte đọc mỗi step — GEMM tính logits vẫn phải đọc đủ ma trận. VRAM 18 GB vốn đã thừa.
- **Đừng hạ conv xuống FP8 thay vì INT4 để né bản vá.** Vấn đề là layer không có quant method nào, không phải định dạng — FP8 cũng lỗi nạp y hệt.
- **Đừng gói nhiều thay đổi vào một lần nộp.** Bản 56.38 đổi 6 thứ, mất 3 điểm, không biết do đâu.
- **Đừng đụng vào bản 59.32** — đó là fallback an toàn.

</details>
