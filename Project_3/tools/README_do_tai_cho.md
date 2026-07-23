# Đo thế nào khi KHÔNG có H200

**Không ai trong đội có H200.** Thứ duy nhất chạm được MiG H200 là **Portal của BTC**.
Vậy nên chia việc như sau — và hoá ra vẫn đủ dùng.

| Câu hỏi | Đo ở đâu |
|---|---|
| TTFT/TPOT thật trên MiG là bao nhiêu? | **Portal** → `tools/infer_tpot.py` |
| Config có khởi động được không? | RTX 3060 |
| Checkpoint requant có nạp được không? | RTX 3060 |
| cudagraph có capture thật hay fallback? | RTX 3060 (đọc log) |
| Độ chính xác tụt bao nhiêu (GPQA)? | RTX 3060 |

---

## 1. Portal là thiết bị đo — `tools/infer_tpot.py`

Portal trả `tbt_median_ms` làm tròn **số nguyên** ("4 ms" có thể là 3.50 hoặc 4.49 — chênh gần 7 điểm). Nhưng `final_score` có **2 chữ số thập phân** và ta biết chính xác công thức chấm. Đảo ngược ra được TPOT.

```bash
python3 tools/infer_tpot.py                                    # bảng các lần đã nộp
python3 tools/infer_tpot.py --score 61.20 --p50 55 --p95 79 --fail 6   # lần nộp mới
```

Kết quả với dữ liệu hiện có:

| Submission | score | **TPOT hiệu dụng** |
|---|---|---|
| `59.32-fp8` (nền) | 59.32 | **4.125 ms** |
| `T1-chunk4096` | 58.44 | **4.152 ms** (+0.027) |

Độ phân giải ~**0.03 ms**. ⇒ **Mỗi lần nộp giờ là một phép đo tử tế**, không còn "nộp mù".

**Cách dùng:** sau MỖI lần nộp, chạy script với 4 số Portal trả về (`final_score`, `ttft_p50_ms`, `ttft_p95_ms`, `failed_count`), ghi TPOT hiệu dụng vào `OPTIMIZATION_ANALYSIS.md` §1. So TPOT chứ đừng so điểm — TPOT tách được ảnh hưởng của TTFT và của số fail.

### Mục tiêu cần đạt (giữ nguyên TTFT 55 ms)

| TPOT | còn 6 fail | vá hết fail |
|---|---|---|
| 2.0 ms | 77.3 | 78.4 |
| 1.8 ms | 79.2 | **80.4** |
| 1.5 ms | 82.3 | **83.5** |
| 1.2 ms | 85.4 | 86.7 |

⇒ **84 điểm = TPOT ~1.35 ms + vá 6 fail.** Không cần đụng tới TTFT.

---

## 2. RTX 3060 là CỔNG KIỂM TRA, không phải thiết bị đo

Mỗi lần nộp một config chết (không khởi động được) là mất trắng một lượt. Chạy thử trên 3060 trước.

```bash
export MODEL_DIR=/duong/dan/LFM2.5-1.2B-Instruct

docker compose -f docker-compose-local-B-cudagraph.yml up -d
docker compose -f docker-compose-local-B-cudagraph.yml logs | grep -iE "cudagraph|capturing|fallback|error"
```

**Cần thấy gì:** dòng log xác nhận vLLM **capture graph ở chế độ FULL**. Nếu log báo fallback về PIECEWISE hoặc bỏ qua layer conv ⇒ **cấu hình T3 vô nghĩa, đừng tốn lượt nộp** — dồn hết sức vào requant INT4.

Smoke-test checkpoint sau requant:

```bash
python3 tools/requant_int4_full.py --verify --out awq_model_int4_conv   # phải <= 0.90 GB
docker run --rm --gpus all -v $PWD/awq_model_int4_conv:/model:ro \
  vllm/vllm-openai:v0.22.1 --model=/model --max-model-len=2048   # phải nạp được
```

### ⚠️ Đừng tin số hiệu năng đo trên 3060
Khác kiến trúc (SM86 vs SM90), khác băng thông, và CPU máy đó **mạnh hơn nhiều** so với 3 core của MiG — nên nó sẽ **che mất** chính phần overhead host mà ta đang cần nhìn. `tools/measure_tpot.py` chỉ dùng để trả lời "có chạy không / có khác gì không", **không** để lấy con số.

---

## 3. Quy trình mỗi vòng lặp

1. Soạn config đổi **đúng 1 biến** so với bản tốt nhất.
2. **3060:** khởi động được? log có đúng như kỳ vọng?
3. **Portal:** nộp.
4. `infer_tpot.py` → TPOT hiệu dụng → ghi vào bảng.
5. Tốt hơn → lấy làm nền mới. Tệ hơn → đóng nhánh, ghi lý do.
