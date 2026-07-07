# Medical Concept Extraction Pipeline — Plan v3

> Bản viết lại từ `plan2`, xử lý các lỗ hổng về điểm số, reproducibility và
> kế hoạch dữ liệu. Nguyên tắc chủ đạo: **deterministic, offline, reproducible,
> data-driven**.

---

## 0. Nguyên tắc thiết kế (rút ra từ luật + công thức chấm)

1. **Deterministic first.** Candidate chấm bằng mã chính xác (Jaccard) → ưu tiên
   pipeline tất định, hạn chế generation nhiễu. Agentic/MCP tool-calling **không dùng**.
2. **Offline & reproducible.** BTC dựng lại code trên private test, có thể **không có
   internet**. Mọi thứ (KB, weights, seed) đóng gói kèm, inference không gọi mạng.
3. **Data-driven.** Luật *yêu cầu* tạo thêm dữ liệu. Được phép dùng **LLM ngoài để
   TẠO/LABEL DATA** (ngoài lời giải chính); chỉ **pipeline inference** mới bị giới hạn
   self-host ≤9B.
4. **Text = substring nguyên văn của input.** Theo JSON mẫu Vòng 1, `text` là chuỗi con
   đúng nguyên bản → dùng NER tagger trả offset, không generation/normalize.

---

## 1. Phân tích điểm số → thứ tự ưu tiên

```
final = 0.3·text(WER) + 0.3·assertions(Jaccard) + 0.4·candidates(Jaccard weighted)
```

| Thành phần | Trọng số | Rủi ro lớn nhất | Ưu tiên |
|---|---|---|---|
| candidates | 0.40 | Sai version KB / chọn sai SET mã | 🔴 Cao nhất |
| text (WER) | 0.30 | Span sai biên / thừa-thiếu entity / lệch chữ | 🔴 Cao |
| assertions | 0.30 | Bỏ sót isHistorical/isNegated/isFamily | 🟠 |
| **type** (ẩn) | — | **Đúng text sai type → tính 2 lần, cả 2 = 0đ trên CẢ 3 metric** | 🔴 Chí mạng |

⇒ **Type chính xác là nền móng**: một span → đúng một type. Đây là lý do bỏ kiến trúc
5-extractor-song-song, thay bằng **một joint tagger**.

---

## 2. Kiến trúc tổng thể

```
                      Clinical Note (input_text, giữ nguyên offset)
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                                 
   Stage 1: JOINT NER TAGGER  (encoder BIO, fine-tuned)
     → spans + type + char offset  (1 span = 1 type, không overlap)
             │
             ├───────────────► Stage 5: POSITION (lấy trực tiếp từ span)
             │
     ┌───────┴────────┐
     ▼                ▼
Stage 3:          Stage 4: CANDIDATE LINKER (offline)
ASSERTION           4a. KB build (RxNorm SCD + ICD-10 VN)  ← 1 lần
(rule + model)      4b. normalize → retrieve → chọn SET mã
 (THUỐC/CHẨN_ĐOÁN/   (THUỐC→RxNorm, CHẨN_ĐOÁN→ICD-10)
  TRIỆU_CHỨNG)
     │                ▼
     └──────┬─────────┘
            ▼
   Stage 6: MERGE → validate schema → N.json
```

Orchestration: **LangGraph làm workflow engine** (state, fan-out/in, retry, logging).
Không ReAct, không agent tự quyết.

---

## 3. Stage 0 — Eval harness + Dev set (LÀM TRƯỚC TIÊN)

Không có thước đo cục bộ thì tối ưu mù. Trước khi code pipeline:

- **Scorer offline — cài ĐÚNG từng chi tiết công thức:**
  - `final = 0.3·text_score + 0.3·assertions_score + 0.4·candidates_score`.
  - **Bước ALIGN (ẩn nhưng bắt buộc):** ghép mỗi entity dự đoán với entity gold để biết
    `pred(k)`↔`gt(k)`. Khoá ghép suy từ note đề: **(text, type)** — khớp text nhưng **lệch
    type → coi như concept THỪA**, tính 2 lần, **cả 2 lần = 0đ trên cả 3 metric**. Entity
    pred không ghép được với gold nào cũng bị coi là thừa (0đ).
  - `text_score = Σ_i (1 − WER(i)) / len(test)`.
  - `assertions_score = Σ_i J_assertions(i) / len(test)`.
  - `candidates_score` = **weighted average toàn cục của Jaccard per-candidate**:
    `Σ_i Σ_k Jaccard(gt(k),pred(k))·(len(gt(k))+1)  /  Σ_i Σ_k (len(gt(k))+1)`
    → Jaccard tính **theo từng candidate k** (từng THUỐC/CHẨN_ĐOÁN), rồi trung bình có
    trọng số `(len(gt(k))+1)`; concept nhiều mã gold nặng ký hơn, `+1` để concept 0-mã vẫn
    có trọng số.
  - **Luật empty-set cho `J_X(i)`:** gt rỗng & pred rỗng → **1**; gt rỗng & pred≠rỗng →
    **0**; còn lại → `|gt∩pred| / |gt∪pred|`.
  - **Đơn vị của Jaccard:** trên **tập mã** (candidates) / **tập nhãn assertion**, không
    phải trên text.
- **Dev set (gold):** ~50–100 doc được label tay chất lượng cao (hoặc LLM-label rồi
  người soát). Dùng để chấm & chọn ngưỡng. **Đây là tài sản quan trọng nhất.**
- **Anchor set:** trích sẵn từ đề để verify KB (mục 4a):
  - RxNorm: `amlodipine 10 mg→308135`, `aspirin 81mg→243670`, `metoprolol succinate xl
    50mg→866436`, `guaifenesin→392085`, `nystatin 5ml→7597`, `acetaminophen 325-650mg→
    313782`, `pravastatin 40mg→904475`, `docusate 100mg→1099279`, `senna 8.6mg→312935`,
    `clonazepam 0.5mg→197527`, `clonazepam 1.5mg→197528`, `Chlorpheniramine 0.4→360047`,
    `Capsaicin 0.38→1660761`.
  - ICD-10: `bệnh trào ngược dạ dày-thực quản → K21.0, K21.9`.

---

## 4. Kế hoạch DỮ LIỆU & huấn luyện (xuyên suốt — plan2 thiếu hẳn)

Nguồn gốc data: gần như chắc là **i2b2/n2c2 2010 (concept + assertion) dịch sang tiếng
Việt** (dòng thuốc tiếng Anh, cấu trúc bệnh án Mỹ). Tận dụng:

- **Mượn nhãn i2b2:** assertion i2b2 (present/absent/possible/hypothetical/conditional/
  **associated_with_someone_else**) map về `isNegated`/`isHistorical`/`isFamily`; concept
  (problem/test/treatment) hỗ trợ khởi tạo NER.
- **Silver labeling bằng LLM ngoài (HỢP LỆ vì chỉ dùng tạo data):** dùng LLM mạnh offline
  (Claude/GPT-4/Qwen-72B...) auto-label một corpus ghi chú lâm sàng tiếng Việt →
  spans+type+assertions+candidate. Đây là **distillation** xuống model ≤9B.
- **Synthetic augmentation:** sinh biến thể (viết tắt, lỗi chính tả, double-space, dính
  chữ) mô phỏng nhiễu thực tế thấy trong input.
- **Fine-tune:**
  - NER tagger (encoder) trên silver+gold.
  - Assertion classifier (encoder) trên nhãn i2b2 + silver.
  - (Tùy chọn) LLM ≤9B cho bước chọn SET candidate (mục 4b).
- **Gold dev giữ riêng** để chấm, không train lên nó.

> Lưu ý luật: LLM ngoài chỉ xuất hiện ở **khâu tạo data offline**, **không** nằm trong
> pipeline inference nộp bài.

---

## 5. Stage 1 — Joint NER Tagger (thay 5 extractor)

- **Model:** encoder token-classification fine-tuned. Ứng viên self-host:
  **PhoBERT-large (~370M)** hoặc **XLM-RoBERTa-large (~550M)** (đa ngữ, hợp vì input trộn
  Việt–Anh).
- **Nhãn BIO cho 5 type:** THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM,
  KẾT_QUẢ_XÉT_NGHIỆM.
- **Vì sao joint thay vì 5 song song:** một token → đúng một nhãn ⇒ **không thể vừa là
  CHẨN_ĐOÁN vừa là TRIỆU_CHỨNG** ⇒ triệt tiêu bẫy sai-type ×2 ở tầng kiến trúc; không cần
  bước reconcile chồng lấn.
- **Offset & duplicate:** span trả char offset trực tiếp → position "miễn phí" và
  **occurrence-aware** (mỗi mention một span riêng, giữ trùng lặp như gold).
- **Doc dài:** chunk theo cửa sổ có overlap (encoder ~256–512 token), map offset về input
  gốc.
- **Đầu ra:** `{text (substring nguyên văn), type, start, end}`.

---

## 6. Stage 3 — Assertion (hybrid rule + model)

Chỉ cho **THUỐC / CHẨN_ĐOÁN / TRIỆU_CHỨNG**. Chạy song song với Stage 4.

- **Input:** span + **cửa sổ ngữ cảnh cục bộ** quanh span (không nhồi cả doc dài).
- **Rule layer (high-precision cues):**
  - `isNegated`: "không", "chưa", "không có", "loại trừ", "âm tính"...
  - `isHistorical`: "tiền sử", "trước khi nhập viện", "đã từng", "tiền căn", mục
    "Thuốc trước khi nhập viện", "các đợt tương tự trước đây"...
  - `isFamily`: "bố", "mẹ", "anh/chị/em", "gia đình", "người nhà", "di truyền"...
- **Model layer:** encoder multi-label (3 nhãn độc lập) fine-tuned trên i2b2+silver, xử lý
  ca ngữ cảnh phức tạp mà rule bỏ sót.
- **Kết hợp:** rule bắt ca rõ (precision cao), model bù ca mờ → tối ưu Jaccard assertion.

---

## 7. Stage 4 — Candidate Linker offline (40% điểm — chi tiết nhất)

### 4a. Build KB (một lần, đóng gói kèm submission)
- **RxNorm:** tải **NLM RxNorm Full Release (RRF)** (tài khoản UMLS miễn phí). Lọc term
  type **SCD/SBD** (mức ingredient+strength+dose form — đúng granularity của mã anchor).
- **ICD-10:** ưu tiên **ICD-10 tiếng Việt của Bộ Y tế** (có tên bệnh tiếng Việt → match
  thẳng CHẨN_ĐOÁN tiếng Việt); bổ sung WHO ICD-10 để phủ mã.
- **Verify version bằng anchor set (Stage 0):** build xong phải tái tạo đúng
  `amlodipine 10 MG Oral Tablet = 308135`, `K21.9 = trào ngược...`. Lệch → đổi
  release/edition. Đây là cách khớp version với BTC mà không cần hỏi.
- **Index:** cho mỗi KB dựng **hybrid**: (a) lexical (BM25/normalized string) +
  (b) embedding (self-host **BGE-m3 / E5-multilingual**).

### 4b. Link + chọn SET candidate (lúc inference)
- **THUỐC:** parse `ingredient + strength + dose form`, **bỏ route/frequency**
  ("po daily", "q6h:prn") → truy vấn RxNorm SCD.
- **CHẨN_ĐOÁN:** normalize cụm tiếng Việt → retrieve trên ICD-10 VN.
- **Chọn SET (điều khiển 40% điểm):** retrieve top-k → **rerank** (cross-encoder hoặc LLM
  ≤9B select) → **ngưỡng calibrate trên dev** để quyết định trả **1 hay nhiều mã**
  (vd `K21.0`+`K21.9`). Thừa mã làm tụt Jaccard (mẫu số union), thiếu mã cũng tụt →
  cân bằng bằng dev.
- **Không emit candidate** cho type khác THUỐC/CHẨN_ĐOÁN (giữ empty đúng luật).

---

## 8. Stage 5 — Position

Lấy **trực tiếp `[start, end]` từ span của Stage 1** (đã 0-indexed theo char trên input
gốc). **Bỏ hẳn `str.find`** → hết lỗi mention trùng lặp & generation drift.

---

## 9. Stage 6 — Merge & Final JSON

- Gộp: NER span + assertions + candidates + position.
- **Validate schema** trước khi ghi: đúng field, type ∈ 5 nhãn, assertions ⊂ {3 giá trị}
  và chỉ ở đúng 3 loại, candidates chỉ ở THUỐC/CHẨN_ĐOÁN, position hợp lệ.
- Ghi `output/N.json` (list dict).

---

## 10. Danh mục model & ngân sách

| Vai trò | Model (self-host) | ~Params | Ghi chú |
|---|---|---|---|
| NER tagger | PhoBERT-large / XLM-R-large | 0.37–0.55B | inference |
| Assertion | encoder multi-label | ~0.3–0.55B | inference |
| Embedding linker | BGE-m3 / E5-multilingual | ~0.56B | inference |
| Rerank/select (tùy chọn) | LLM ≤9B (Qwen2.5-7B...) | ≤9B | chỉ ca khó |
| Data labeling | LLM ngoài mạnh | — | **chỉ offline tạo data** |

⚠️ **Cần xác nhận BTC:** giới hạn 9B là **per-model hay tổng**? Ảnh hưởng việc dùng nhiều
model nhỏ song song.

---

## 11. Rủi ro & câu hỏi cần chốt với BTC

1. **9B per-model hay tổng?** (mục 10).
2. **ICD-10 edition nào** BTC dùng cho ground truth (WHO / VN-BYT / CM)? — đang dò bằng
   anchor, nhưng hỏi được thì chắc hơn.
3. **RxNorm release tháng nào** (mã có thể đổi giữa các version).
4. **WER tính thế nào trên list entity** (concat toàn bộ text / align rồi WER từng
   entity)? — ảnh hưởng chiến lược precision/recall của NER.
5. **Khoá ALIGN pred↔gold** chính xác là gì (exact `(text,type)` / có tolerance WER /
   dựa position)? — quyết định cách scorer đếm concept thừa-thiếu.
6. **`J_candidates` gộp per-candidate hay pool cả sample** (đọc công thức nghiêng về
   per-candidate weighted, nhưng nên xác nhận).

---

## 12. Lộ trình (milestones)

1. **M1 — Scorer + dev gold** (Stage 0). Không có cái này không đo được gì.
2. **M2 — KB offline + anchor verify** (Stage 4a).
3. **M3 — Baseline:** NER tagger (silver) + linker lexical + rule assertion → đo dev.
4. **M4 — Fine-tune** NER & assertion trên silver+i2b2; thêm embedding rerank.
5. **M5 — Calibrate SET candidate**, tối ưu ngưỡng theo dev.
6. **M6 — Đóng gói reproducible:** weights + KB index + seed + README, chạy offline
   end-to-end trên `input/` → `output/`.

---

## Tóm tắt thay đổi so với plan2

| plan2 | plan3 |
|---|---|
| 5 extractor LLM song song | 1 **joint NER tagger** (triệt tiêu bẫy sai-type) |
| LLM generation + `str.find` | **encoder BIO** trả offset trực tiếp |
| Stage 4 = "API/Database" 1 dòng | **KB offline + hybrid retrieval + chọn SET** (chi tiết) |
| Không có data/fine-tune | **Silver labeling + distillation + i2b2**, LLM ngoài chỉ tạo data |
| Không có eval | **Stage 0: scorer + dev gold + anchor** làm trước |
| Không xử lý assertion dài / duplicate | **cửa sổ cục bộ** + **occurrence-aware span** |
| Reproducibility mờ | **Offline, đóng gói KB+weights**, xác nhận version bằng anchor |
