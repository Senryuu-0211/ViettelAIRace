# CLAUDE.md — Project 2: Medical Knowledge Retrieval

> Quy tắc giao tiếp & môi trường: xem hub `.claude/CLAUDE.md` ở gốc repo.
> File này chỉ chứa phần đặc thù bài toán y khoa.

---

## 1. Bài toán
Xử lý **văn bản y khoa tự do tiếng Việt** (ghi chú bác sĩ, giấy xuất viện, kết quả XN, EHR) để **phát hiện + chuẩn hóa khái niệm y tế** (NER + entity linking) và **suy luận ngữ cảnh** (phủ định / người nhà / tiền sử).
KB chuẩn: **ICD-10** (chẩn đoán) và **RxNorm** (thuốc). Cuộc thi: Viettel AI Race — Vòng 1 (02/07–30/07/2026).

## 2. Input / Output
- Input: `input/N.txt` (N=1..100), free-form, mỗi file **nhiều khái niệm**, có viết tắt/lỗi chính tả.
- Output: `output/N.json` — **list dict**. **`position` = offset KÝ TỰ, 0-indexed, trên input GỐC** (không normalize trước khi tính — nguồn lỗi phổ biến nhất).

| Trường | Mô tả |
|---|---|
| `text` | Cụm từ trong input (**substring nguyên văn**) |
| `position` | `[start, end]` ký tự 0-indexed |
| `type` | 1 trong 5 nhãn |
| `assertions` | `isNegated` / `isFamily` / `isHistorical` — **chỉ** CHẨN_ĐOÁN, THUỐC, TRIỆU_CHỨNG |
| `candidates` | Mã chuẩn — **chỉ** CHẨN_ĐOÁN (ICD-10) và THUỐC (RxNorm) |

**5 nhãn:** `TRIỆU_CHỨNG` · `TÊN_XÉT_NGHIỆM` · `KẾT_QUẢ_XÉT_NGHIỆM` (giá trị+đơn vị) · `CHẨN_ĐOÁN` → ICD-10 · `THUỐC` → RxNorm.

```json
{"text":"amlodipine 10 mg po daily","type":"THUỐC","candidates":["308135"],"assertions":["isHistorical"],"position":[58,83]}
```

## 3. Metric
```
final = 0.3·text(WER) + 0.3·assertions(Jaccard) + 0.4·candidates(Jaccard weighted)
```
- `candidates_score` = trung bình có trọng số của Jaccard **per-candidate**, weight = `len(gold)+1`.
- Jaccard rỗng: gold rỗng & pred rỗng → 1; gold rỗng & pred≠rỗng → 0.
- 🔴 **Sai `type` = khái niệm bị tính 2 lần, cả 2 đều 0đ trên CẢ 3 metric** ⇒ phân loại type là nền móng.
- Scorer align entity theo **(text, type)** — xem `LLM-solution/scorer.py`.

## 4. Ràng buộc
- **Không dùng API LLM ngoài trong pipeline inference.** Self-host, model ≤ **9B params**.
- **Được dùng LLM ngoài để TẠO/LABEL DATA** (ngoài lời giải chính) — luật cho phép.
- Luật *giả định/khuyến khích* train ("cần tạo thêm dữ liệu nhằm huấn luyện mô hình") nhưng **KHÔNG bắt buộc** — prompt-only vẫn hợp lệ.
- Nộp `output.zip` (1.json…100.json). Top ~15 phải nộp code + data + weights + README để BTC **dựng lại chấm private test** ⇒ **reproducibility là bắt buộc**.

---

## 5. Trạng thái hiện tại

**2 hướng đang song song (chưa reconcile):**
| Hướng | Nhánh | Nội dung |
|---|---|---|
| **Neo4j KG** (của bạn Mr. Senryuu) | `develop` | `LLM-solution/` + `icd10_knowledge_graph.json` + `neo4j_linker.py` |
| **Hybrid retrieval** (Claude) | `develop-hybrid` | ICD linker lexical+embedding trên KB phẳng |

**Đã làm được (hướng hybrid):**
- **KB ICD-10 TT06 sạch:** crawl từ `icd.kcb.vn` — API thật `https://ccs.whiteneuron.com/api/ICD10_TT06/`, duyệt `childs/<model>?id=<id>` (chapter→section→type→disease). Kết quả **18.237 mã + 2.090 category** kèm tên tiếng Việt + **SET expansion** (K21 → {K21.0, K21.9}). Crawler: `tools/crawl_icd10_tt06.py`.
- **ICD linker hybrid**: lexical (rapidfuzz) + embedding cosine + fuse min-max + mở rộng theo category. **Mặc định `USE_EMBEDDING=0` (lexical-only)** — vì embedding trên CPU quá chậm (7-11s/query) và **không cải thiện** (lexical 66.7% ≈ hybrid).
- **Embedder**: gọi thẳng Ollama `/api/embed` (batch). **KHÔNG dùng `langchain_ollama` cho embedding** (bản 1.1.0 lỗi endpoint `/tokenize`).
- **Scorer** đúng công thức (`scorer.py`).

**Model & gotcha lớn nhất:**
- Dùng **`qwen3.5:4b-q4_K_M`** qua Ollama.
- 🔴 **BẮT BUỘC `reasoning=False` trong ChatOllama.** Không tắt → qwen3.5 chèn token "thinking" → **0 entity + 18 phút/file**. Tắt rồi: ~2.6 phút/file.
- Baseline đo được: **FINAL ~0.50** trên gold drug-list (`tools/eval_druglist.py`).

---

## 6. Còn nợ
- **Chưa có gold dev set thật** (mới chấm được trên 1 ví dụ drug-list lấy từ đề) → cần label vài file `input/` (qua UI hoặc tay). **Đây là điều kiện tiên quyết để tune prompt tử tế** — tune trên 1 mẫu sẽ overfit (đã thử, bị regress).
- **RxNorm linker còn bug**: ép lọc `SCD+SBD` → mất mã mức ingredient (vd `7597`). Verify bằng 13 anchor lấy từ đề.
- Lỗi prompt đã biết: cụm "thuốc + *điều trị X*" bị gán nhầm `TRIỆU_CHỨNG`.
- Quyết định team: chọn hybrid hay Neo4j; có train (fine-tune NER + assertion encoder) hay giữ prompt-only.

## 7. Lưu ý khi code
- `position`: luôn tính offset trên input **gốc** (giữ nguyên dấu, khoảng trắng lạ, xuống dòng).
- `text` phải là **substring nguyên văn** → ưu tiên trích theo span, tránh để LLM "viết lại".
- Không dedupe mention trùng: mỗi lần xuất hiện là **1 entity riêng** (gold giữ trùng lặp).
