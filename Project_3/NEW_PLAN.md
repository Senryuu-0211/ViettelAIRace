# LFM2.5-1.2B-Instruct Serving Optimization Challenge — Phase (Online Round)

## 1. Tổng quan

Triển khai và tối ưu một LLM inference server cho mô hình **LiquidAI/LFM2.5-1.2B-Instruct**, xử lý một workload trace **multi-turn** mô phỏng traffic production:

- **70 hội thoại** đến theo phân phối **Poisson**
- Tổng **330 request được chấm** (sau **15 hội thoại primer** khởi động — không tính điểm)
- Context input tối đa **~4K token** (~12K ký tự)
- Output tối đa **200 token**

Trong vòng online, mục tiêu là tối đa hóa **ERS** (điểm độ trễ). **Accuracy Gate chỉ chạy sau khi vòng online kết thúc**, trên tối đa **5 submissions** do đội tự chọn.

---

## 2. Hạ tầng & môi trường đánh giá

| Thuộc tính | Giá trị |
|---|---|
| Hardware | 1 instance MiG H200 (18GB VRAM, 3 Core CPU, 8GB RAM) — cấp tự động mỗi lượt chấm |
| OS & Driver (host) | Ubuntu 24.04 LTS, NVIDIA driver 590.x (CUDA 13.x) |
| Model | LiquidAI/LFM2.5-1.2B-Instruct |
| Weights | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct |

---

## 3. Cách tính điểm

### 3.1 ERS (Effective Request Score) — chấm trong vòng online

```
ERS = (1/N) × Σ S_request,i     ,  ERS ∈ [0, 1]      (N = tổng số request)

S_request = 0                          nếu lỗi, timeout, hoặc trả về 0 token
S_request = w·s_ttft + (1−w)·s_tpot    nếu xử lý thành công

s_ttft = [ clamp( (C_ttft − TTFT) / (C_ttft − F_ttft), 0, 1 ) ] ^ γ
s_tpot = [ clamp( (C_tpot − TPOT_mean) / (C_tpot − F_tpot), 0, 1 ) ] ^ γ
```

**Tham số cấu hình (Phase này):**

| Ký hiệu | Ý nghĩa | Giá trị |
|---|---|---|
| F_ttft | Floor của TTFT | **10 ms** |
| C_ttft | Ceiling của TTFT | **400 ms** |
| F_tpot | Floor của TPOT | **1 ms** |
| C_tpot | Ceiling của TPOT | **10 ms** |
| γ | Hệ số lũy thừa | 2 |
| w | Trọng số của TTFT | 0.5 |

> ⚠️ Lưu ý: ngưỡng Floor/Ceiling ở round này **chặt hơn nhiều** so với round trước (TTFT ceiling 400ms so với 1500ms trước đây; TPOT ceiling chỉ 10ms so với 45ms trước đây) — do model nhỏ hơn (1.2B) và context ngắn hơn nhiều (~4K token so với ~20K token).

### 3.2 Accuracy Gate — chạy SAU vòng online (khác cơ chế trước)

- **Không** chấm GPQA trên từng lượt nộp online.
- Sau khi vòng online kết thúc: đội chọn thủ công **tối đa 5 submissions** tốt nhất.
- BTC lần lượt: (1) hậu kiểm tính hợp lệ phương án → (2) dựng endpoint và chạy **GPQA full**.

```
Δ = Accuracy_baseline − Accuracy_submission     (Accuracy_baseline mặc định = 0.4, đo bằng BF16 gốc)

f(Δ) = 1.0                        nếu Δ ≤ 0.10
f(Δ) = 1.0 − (Δ − 0.10) / 0.06    nếu 0.10 < Δ < 0.16
f(Δ) = 0.0                        nếu Δ ≥ 0.16
```

### 3.3 Điểm cuối mỗi submission hợp lệ

```
Score = 100 × ERS × f(Δ)
```

- `ERS` lấy từ **lần chấm online** của đúng bài đó (không chấm lại).
- `f(Δ)` chỉ có **sau bước GPQA post-online**.
- **Điểm đội = Score tốt nhất** trong các bài còn hợp lệ (trong số 5 bài đã chọn).

---

## 4. Không gian tối ưu

> ⚠️ **Chỉ được phép dùng serving framework vLLM** cho bài thi này (khác round trước — trước đây cho phép chọn vLLM/SGLang/TensorRT-LLM/custom).

- **Quantization**: các kỹ thuật Online Quantization
- **KV Cache & Memory**: Paged Attention để tối đa hóa request đồng thời; KV cache quantization (FP8, INT8); Prefix caching và Semantic caching; Offloading CPU/NVMe
- **Serving & Scheduling**: Dynamic/Continuous batching; Speculative decoding; Memory-aware scheduling
- **System & Runtime**: Custom CUDA/Triton kernels; Fused attention kernels (FlashAttention, FlashInfer); Tối ưu memory layout và CUDA Graphs

---

## 5. Nộp bài & tài nguyên

### Quy trình

1. **Develop & Package**: phát triển code, tối ưu hệ thống, đóng gói thành Docker Image
2. **Push Image**: đẩy Docker Image lên Docker Hub (Public)
3. **Submit**: gửi `docker-compose.yml` qua Portal BTC (khai báo đúng image + lệnh thực thi)
4. **Automated Evaluation**: hệ thống pull image, dựng container trên MiG H200, healthcheck, chạy benchmark **ERS** (không chạy GPQA mỗi lượt)
5. **Leaderboard**: cập nhật theo ERS
6. **Sau vòng online**: đội chọn tối đa 5 submissions → BTC hậu kiểm hợp lệ → chạy GPQA full (lm_eval / bench-gpqa-diamond.sh) → chốt Score

### Tài nguyên

- File trace công khai (bản lược — chỉ arrival + số token in/out, **không có prompt thật**): `trace_grading_public.jsonl`. BTC giữ bản đầy đủ (có prompt thật) để chấm.
- Docker image baseline: [vllm/vllm-openai:v0.22.1](https://hub.docker.com/layers/vllm/vllm-openai/v0.22.1/images/sha256-55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5)

### File `docker-compose.yml` mẫu

```yaml
services:
  model:
    image: vllm/vllm-openai:v0.22.1
    entrypoint:
      - python3 #Don't change this to vllm-server
      - -m  #Don't change this to vllm-server
      - vllm.entrypoints.openai.api_server #Don't change this to vllm-server
    command:
      - --model=/model #Don't change this to vllm-server
      - --served-model-name=LFM2.5-1.2B-Instruct #Don't change this to vllm-server
      - --host=0.0.0.0 #Don't change this to vllm-server
      - --port=8000 #Don't change this to vllm-server
      - --max-model-len=32768
      - --gpu-memory-utilization=0.95
      - --tensor-parallel-size=1
      - --enable-prefix-caching
    ports:
      - "8000:8000"
    shm_size: "2g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 6. Quy định & phòng chống gian lận

Áp dụng nghiêm ngặt nguyên tắc Anti-Cheating tại tab Tổng quan. **Yêu cầu cốt lõi**: serving LLM trung thực trên tài nguyên GPU BTC cấp phát.

**Nghiêm cấm:**
- ❌ Pre-bake, hardcode kết quả, cơ chế dual-path hoặc lách luật (gaming) phương pháp đo lường
- ❌ Thực hiện lệnh gọi mạng bên ngoài
- ❌ Can thiệp trái phép vào tokenizer hoặc weights của model
- ❌ Tráo đổi Docker image **sau khi đã nộp bài**

**Quy trình chấm & hậu kiểm:**
- Trong vòng Online: chỉ chấm tự động dựa trên ERS
- Sau vòng thi: đội chọn tối đa 5 submissions → BTC hậu kiểm tính hợp lệ → chạy GPQA full
- BTC bảo lưu quyền hủy kết quả nếu có dấu hiệu gian lận; quyết định xử lý thông báo qua email

---

## 7. Hậu kiểm, Tie-break & Khiếu nại

**Tiêu chí phụ khi điểm số bám sát nhau** (trong biên độ nhiễu đo lường ≤ 1-2 điểm), xếp theo thứ tự ưu tiên:

1. Mức độ suy giảm độ chính xác
2. Chỉ số p95 TTFT
3. Tốc độ sinh văn bản
4. Thời điểm nộp bài (ưu tiên bài nộp sớm hơn)

**Quy trình hậu kiểm & chấm lại:**
- BTC ưu tiên hậu kiểm kỹ các cặp đội có tính cạnh tranh cao (đặc biệt nhóm tranh chấp giải thưởng)
- BTC có quyền chấm lại, lấy điểm trung vị của các lần chạy trên Docker image đã chốt

**Khiếu nại:**
- Trước khi chốt bảng xếp hạng chung cuộc, BTC gửi email thông báo kết quả dự kiến
- Khiếu nại phải gửi trong tối đa **24 giờ** kể từ khi nhận email hoặc khi kết quả Phase được công bố chính thức