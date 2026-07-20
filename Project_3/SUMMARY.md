# LLM Inference Optimization Challenge

## 1. Giới thiệu chung

Cuộc thi mô phỏng trực tiếp thách thức mà các đội ngũ hạ tầng AI doanh nghiệp đang đối mặt: phục vụ mô hình ngôn ngữ lớn (LLM) sao cho đáp ứng đồng thời:

- **Thông lượng cao**
- **Độ trễ thấp**
- **Độ chính xác ổn định**
- **Hiệu quả trên tài nguyên GPU hữu hạn**

Thí sinh sẽ deploy và tối ưu serving stack cho một mô hình LLM (do Ban tổ chức - BTC - chỉ định) trên hạ tầng **NVIDIA H200**, với workload được mô phỏng theo production trace của một hệ thống LLM serving quy mô thực tế.

Bài toán cho phép tự do lựa chọn phương pháp tối ưu — từ quantization, KV cache management, prefix caching, đến custom CUDA kernel và scheduling — để tối đa hoá tỷ lệ request được đáp ứng đúng yêu cầu, trong khi vẫn đảm bảo chất lượng đầu ra.

### Khái niệm chính

| Thuật ngữ | Ý nghĩa |
|---|---|
| **TTFT** (Time-To-First-Token) | Thời gian từ khi gửi request đến khi nhận token đầu tiên |
| **TPOT** (Time-Per-Output-Token) | Thời gian giữa 2 token liên tiếp trong stream output |

---

## 2. Mục tiêu bài toán

Đây là bài toán **LLM serving optimization có ràng buộc về chất lượng**. Mục tiêu:

> Tối đa hóa **Effective Request Score (ERS)** trên toàn bộ workload trace cố định do BTC phát hành, đồng thời vượt qua bài kiểm tra chất lượng (**Accuracy Gate**).

- **Latency Bounds**: điểm được nội suy liên tục dựa trên cận dưới (Floor – lý tưởng) và cận trên (Ceiling – không thể chấp nhận) của độ trễ. Ngưỡng cụ thể công bố theo từng vòng.
- **Accuracy Gate**: độ chính xác đo qua bài test **GPQA Diamond**, không được suy giảm quá ngưỡng quy định theo từng vòng (xác định trước dựa trên baseline).

> Lưu ý: tất cả các đội chạy cùng một file trace với cùng arrival timestamp.

---

## 3. Cách tính điểm (công thức chung)

### 3.1 ERS (Effective Request Score)

Điểm ERS là trung bình cộng điểm của tất cả N request trong file trace:

```
ERS = (1/N) × Σ S_request,i     ,  ERS ∈ [0, 1]
```

Điểm từng request:

```
S_request = 0                          nếu lỗi, timeout, hoặc trả về 0 token
S_request = w·s_ttft + (1−w)·s_tpot    nếu xử lý thành công
```

Điểm thành phần độ trễ (nội suy giữa Floor F và Ceiling C):

```
s_ttft = [ clamp( (C_ttft − TTFT) / (C_ttft − F_ttft), 0, 1 ) ] ^ γ

s_tpot = [ clamp( (C_tpot − TPOT) / (C_tpot − F_tpot), 0, 1 ) ] ^ γ
```

**Tham số:**
- `F_ttft`, `F_tpot`: Floor (độ trễ ≤ mức này → điểm tối đa, s = 1)
- `C_ttft`, `C_tpot`: Ceiling (độ trễ ≥ mức này → điểm 0)
- `w`: trọng số ưu tiên TTFT (0 < w < 1)
- `γ`: hệ số lũy thừa (γ ≥ 1) quy định độ dốc hàm phạt
- `clamp(x, 0, 1)`: giới hạn x trong đoạn [0, 1]

### 3.2 Accuracy Gate — GPQA Diamond

Đánh giá độc lập bằng bộ 100 câu hỏi cố định từ GPQA Diamond.

```
Δ = baseline_accuracy − GPQA_accuracy_của_đội
```

Hàm phạt suy giảm chất lượng (piecewise linear, f(Δ) ∈ [0, 1]):

```
f(Δ) = 1.0                        nếu Δ ≤ 0.10
f(Δ) = 1.0 − (Δ − 0.10) / 0.06    nếu 0.10 < Δ < 0.16
f(Δ) = 0.0                        nếu Δ ≥ 0.16
```

### 3.3 Công thức điểm tổng

```
Score = 100 × ERS × f(Δ)
```

---

## 4. Mô hình sử dụng (chung cho các vòng)

| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | Dense Transformer |
| Precision gốc | BF16 (native release) |
| License | Apache 2.0 |
| Nguồn weights | HuggingFace Hub (hash cố định công bố theo từng vòng) |

---

## 5. Phương pháp tối ưu được phép

- **KV Cache Optimization**: KV cache quantization (FP8, INT8), KV cache offloading (CPU/NVMe), prefix caching, semantic caching, Paged Attention, memory-aware scheduling
- **Serving & Scheduling**: Dynamic/continuous batching, speculative decoding (draft model hoặc self-speculative), disaggregated prefill/decode serving
- **System-Level**: Custom CUDA/Triton kernels, fused attention kernels (FlashAttention, FlashInfer...), NCCL optimization cho NVLink topology, CUDA Graphs, memory layout optimization
- **Runtime & Compiler**: vLLM, SGLang, TensorRT-LLM, Transformers, hoặc custom runtime; tùy chỉnh tensor/pipeline parallelism; overlap communication/computation

---

## 6. Môi trường đánh giá chuẩn hóa (chung)

- Hạ tầng: NVIDIA H200 GPU
- Hệ điều hành: Ubuntu 22.04 LTS
- CUDA: 12.x
- GPU Driver: phiên bản tiêu chuẩn do BTC cung cấp

---

## 7. Rule & Anti-Cheating

Nghiêm cấm tuyệt đối:

- ❌ Hardcode đáp án của probe subset trong mã nguồn
- ❌ Pre-compute response cho các request nằm trong trace
- ❌ Gọi network external từ inference server khi đang serving
- ❌ Chỉnh sửa tokenizer của mô hình
- ❌ Thay đổi arrival timestamp của trace hoặc cấu hình concurrency
- ❌ Dùng account khác/phụ để leak hidden trace giữa các đội

**Xử lý vi phạm:**
- Vi phạm → submission bị void
- Vi phạm nghiêm trọng → đội bị loại
- Khiếu nại: gửi BTC trong vòng 24h sau khi công bố kết quả phase tương ứng

---

## 8. Phase 1 — Chi tiết vòng online

### 8.1 Tổng quan

**Nhiệm vụ:** Triển khai và tối ưu một LLM inference server cho mô hình **Qwen/Qwen3.5-2B**, xử lý file trace gồm **120 requests** mô phỏng traffic production. Mục tiêu: tối đa hoá Effective Request Capacity trong khi vượt qua Accuracy Gate.

### 8.2 Hạ tầng & môi trường đánh giá

- Toàn bộ benchmark chạy tự động trên hệ thống BTC
- Thí sinh serve endpoint trên **1 instance MiG H200 (18GB VRAM, 3 Core CPU, 8GB RAM)** cấp phát tự động mỗi lượt chấm
- OS/Driver: Ubuntu 22.04 LTS, CUDA 12.x
- Model: Qwen/Qwen3.5-2B (Dense Transformer, gốc BF16)
- Nguồn weights: HuggingFace Hub (hash cố định do BTC công bố)

### 8.3 Tham số chấm điểm Phase 1

| Ký hiệu | Ý nghĩa | Giá trị |
|---|---|---|
| F_ttft | Floor của TTFT | 100 ms |
| C_ttft | Ceiling của TTFT | 1500 ms |
| F_tpot | Floor của TPOT | 20 ms |
| C_tpot | Ceiling của TPOT | 45 ms |
| γ | Hệ số lũy thừa | 2 |
| w | Trọng số của TTFT | 0.5 |

**Accuracy Gate (Phase 1):**
- baseline_accuracy mặc định = **0.4**
- Công thức Δ và f(Δ) giữ nguyên như mục 3.2

### 8.4 Không gian tối ưu (Phase 1)

- **Quantization**: weight quantization (FP8/F8_E4M3, INT8, INT4, mixed-precision, AWQ, GPTQ); activation quantization, dynamic quantization
- **KV Cache & Memory**: Paged Attention, KV cache quantization (FP8, INT8), prefix caching, semantic caching, offloading CPU/NVMe
- **Serving & Scheduling**: dynamic/continuous batching, speculative decoding (draft model/self-speculative), memory-aware scheduling
- **System & Runtime**: custom CUDA/Triton kernels, fused attention kernels (FlashAttention, FlashInfer), tối ưu memory layout, CUDA Graphs

### 8.5 Quy trình nộp bài (Submission Workflow)

1. **Develop & Package**: phát triển code giải pháp, tối ưu hệ thống, đóng gói thành Docker Image
2. **Push Image**: đẩy Docker Image lên Docker Hub cá nhân/tổ chức (Public)
3. **Submit**: truy cập Portal BTC, gửi file `docker-compose.yml` (khai báo đúng đường dẫn Image trên Docker Hub và lệnh thực thi)
4. **Automated Evaluation**: hệ thống tự động pull Image, dựng container trên 1 instance MiG H200 (18GB VRAM), healthcheck, chạy benchmark tự động
5. **Leaderboard**: kết quả và log trả về trong ~15 phút; bảng xếp hạng tự động cập nhật

### 8.6 Tài nguyên Phase 1

- File trace: `trace-round1.jsonl`
- Docker image baseline: [vllm/vllm-openai:v0.22.1](https://hub.docker.com/layers/vllm/vllm-openai/v0.22.1/images/sha256-55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5)

### 8.7 File `docker-compose.yml` mẫu

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
      - --served-model-name=Qwen3.5-2B #Don't change this to vllm-server
      - --host=0.0.0.0 #Don't change this to vllm-server
      - --port=8000 #Don't change this to vllm-server
      - --max-model-len=262144
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