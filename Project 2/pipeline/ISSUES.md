# Pipeline Issues

## Output Format

```json
{
  "text": "...",
  "type": "THUỐC|CHẨN_ĐOÁN|TRIỆU_CHỨNG|TÊN_XÉT_NGHIỆM|KẾT_QUẢ_XÉT_NGHIỆM",
  "assertions": ["isNegated"|"isHistorical"|"isFamily"],
  "candidates": ["mã_code"],
  "position": [start, end]
}
```

---

## 1. Entity Extraction

### Prompt hiện tại (`pipeline/prompts/entity_template.py`)

```
Bạn là chuyên gia trích xuất khái niệm y khoa từ ghi chú lâm sàng tiếng Việt.
Đọc ghi chú và trả về TẤT CẢ các khái niệm y khoa, phân loại đúng vào 5 nhóm sau:

1. THUỐC: tên thuốc trong ghi chú lâm sàng. Nếu tên thuốc có các từ bổ nghĩa
   đứng LIỀN KỀ (liều lượng, đường dùng, tần suất) thì gộp tất cả thành 1 entity.
   - VD: "metoprolol 25mg po bid", "amlodipine 10 mg po daily",
     "aspirin 325mg", "Capsaicin 0.38 MG/ML", "prednisone", "atenolol"

2. CHẨN_ĐOÁN: tên bệnh, hội chứng, tình trạng bệnh lý được chẩn đoán.
   - VD: "tăng huyết áp", "bệnh trào ngược dạ dày - thực quản",
     "hội chứng não gan", "tăng lipid máu không đặc hiệu", "xơ vữa động mạch"

3. TRIỆU_CHỨNG: dấu hiệu, triệu chứng lâm sàng, than phiền của bệnh nhân.
   - VD: "đau ngực", "khó thở", "ho đờm xanh", "buồn nôn", "đánh trống ngực",
     "mệt mỏi", "sốt"

4. TÊN_XÉT_NGHIỆM: tên xét nghiệm, chỉ số xét nghiệm, thủ thuật cận lâm sàng.
   - VD: "WBC", "NEUT%", "siêu âm gan mật", "chụp x-quang ngực",
     "ECG", "AST", "ALT", "bilirubin toàn phần"

5. KẾT_QUẢ_XÉT_NGHIỆM: giá trị kết quả xét nghiệm bằng số hoặc mô tả kết quả.
   - VD: "14,43", "76,4", "tăng nhẹ", "âm tính"

QUAN TRỌNG (PHẢI TUÂN THỦ TUYỆT ĐỐI):
- "text" PHẢI là chuỗi con NGUYÊN VĂN từ input, không thêm/bớt/sửa ký tự nào.
  Copy-paste chính xác từ input, giữ nguyên dấu cách, dấu câu, viết hoa/thường.
- Mỗi entity là MỘT cụm từ LIÊN TỤC mô tả MỘT khái niệm y khoa duy nhất.
  KHÔNG tách một khái niệm thành nhiều entity nhỏ (vd: không tách thuốc, không
  tách tên xét nghiệm, không tách triệu chứng).
- KHÔNG gộp các khái niệm khác nhau vào chung một entity.
- Ghi chú trong ngoặc đơn như "(uống hôm nay)", "(trước đây)" KHÔNG phải một phần
  của entity. Chỉ lấy phần chính của khái niệm.
- Mỗi text chỉ thuộc ĐÚNG 1 type. Nếu một cụm vừa có thể là CHẨN_ĐOÁN vừa là
  TRIỆU_CHỨNG, chọn CHẨN_ĐOÁN (chẩn đoán mạnh hơn).
- Trích xuất TẤT CẢ các mention, kể cả trùng lặp (mỗi vị trí trong văn bản là 1 entity).
- Không bịa ra text không có trong input.

Ví dụ minh họa cách xác định ranh giới entity:
Input: "...dùng metoprolol 25mg po bid và atenolol (uống hôm nay) do tăng huyết áp"
Output: [{"text": "metoprolol 25mg po bid", "type": "THUỐC"},
         {"text": "atenolol", "type": "THUỐC"},
         {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN"}]

Trả về CHỈ JSON array, KHÔNG markdown, KHÔNG giải thích:
[{"text": "...", "type": "THUỐC"}, ...]
```

### Vấn đề

| # | Mô tả | Ví dụ |
|---|-------|-------|
| A | Gộp nhiều khái niệm thành 1 entity | `"lo âu mất ngủ"` → 1 entity thay vì 2 (`"lo âu"` + `"mất ngủ"`) |
| B | Sai type: gán CHẨN_ĐOÁN thay vì TRIỆU_CHỨNG | `"táo bón"`, `"lo âu"` → CHẨN_ĐOÁN, gold là TRIỆU_CHỨNG |
| C | Thiếu entity sau `"điều trị X"` | `"ho"`, `"đau nhức"` không được extract |
| D | Không extract đủ duplicate | 2× `"táo bón"`, 2× `"lo âu"` mà output chỉ có 1 lần |
| E | (ĐÃ FIX) Tách thuốc thành nhiều phần | `"metoprolol 25mg po bid"` → giờ là 1 entity |

### Test case: drug list (so sánh gold vs pipeline)

**Input:**
```
Danh sách thuốc trước nhập viện chính xác và đầy đủ.
1. amlodipine 10 mg po daily
2. aspirin 81 mg po daily
...
8. docusate sodium 100 mg po bid điều trị táo bón
9. senna 8.6 mg po bid:prn điều trị táo bón
10. clonazepam 0.5 mg po qam:prn điều trị lo âu
11. clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ
```

| | Gold (19 entity) | Pipeline (15 entity) |
|---|---|---|
| 11 THUỐC | Đủ | Đủ |
| `"ho"` | TRIỆU_CHỨNG | Thiếu |
| `"đau nhức"` | TRIỆU_CHỨNG | Thiếu |
| `"sốt đau"` | TRIỆU_CHỨNG | TRIỆU_CHỨNG |
| `"táo bón"` (×2) | TRIỆU_CHỨNG ×2 | CHẨN_ĐOÁN ×1 |
| `"lo âu"` (×2) | TRIỆU_CHỨNG ×2 | CHẨN_ĐOÁN ×1 |
| `"mất ngủ"` | TRIỆU_CHỨNG | Thiếu (bị gộp vào `"lo âu mất ngủ"`) |
| `"lo âu mất ngủ"` | Không tồn tại trong gold | TRIỆU_CHỨNG (gộp sai) |

---

## 2. Assertion

### Prompt hiện tại (`pipeline/prompts/assertion_batch_template.py`)

```
Bạn là chuyên gia phân tích ngữ cảnh lâm sàng. Cho một danh sách các entity y khoa
cùng với ghi chú lâm sàng (mỗi entity được đánh dấu bằng **...** trong văn bản),
xác định assertion cho TỪNG entity. 

Với mỗi entity, trả về 0-3 assertion trong:
- "isNegated": entity bị PHỦ ĐỊNH (không có, loại trừ, âm tính, chưa)
- "isHistorical": entity thuộc TIỀN SỬ (đã từng, trước đây, tiền căn, trước khi nhập viện)
- "isFamily": entity của NGƯỜI NHÀ (bố, mẹ, anh, chị, em, gia đình, di truyền)

Nếu không có assertion nào -> trả về mảng rỗng [].

QUAN TRỌNG:
- isHistorical và isNegated CÓ THỂ cùng xuất hiện
- isFamily thường đi với isHistorical
- Phân biệt "tiền sử" với "hiện tại"
- Chỉ assertion khi ngữ cảnh hỗ trợ rõ ràng

Trả về CHỈ JSON object với key là entity_text và value là array assertions:
{"entity_text_1": ["isHistorical"], "entity_text_2": [], ...}
```

### Vấn đề

| # | Mô tả | Chi tiết |
|---|-------|----------|
| A | isHistorical không nhất quán | 11 thuốc trong `"Danh sách thuốc trước nhập viện"` → chỉ 3/11 được gán isHistorical, 8/11 không có |
| B | isFamily false-positive | `"táo bón"`, `"lo âu mất ngủ"` bị gán isFamily dù là của bệnh nhân |

### Test case: drug list

| Entity | Gold assertion | Pipeline assertion |
|--------|:---:|:---:|
| amlodipine 10 mg po daily | `["isHistorical"]` | `[]` |
| aspirin 81 mg po daily | `["isHistorical"]` | `[]` |
| metoprolol succinate xl 50 mg po daily | `["isHistorical"]` | `[]` |
| guaifenesin ml po q6h:prn | `["isHistorical"]` | `["isHistorical"]` |
| nystatin oral suspension 5 ml po qid:prn | `["isHistorical"]` | `["isHistorical"]` |
| acetaminophen 325-650 mg po q6h:prn | `["isHistorical"]` | `["isHistorical"]` |
| pravastatin 40 mg po daily | `["isHistorical"]` | `[]` |
| docusate sodium 100 mg po bid | `["isHistorical"]` | `[]` |
| senna 8.6 mg po bid:prn | `["isHistorical"]` | `[]` |
| clonazepam 0.5 mg po qam:prn | `["isHistorical"]` | `[]` |
| clonazepam 1.5 mg po qhs | `["isHistorical"]` | `[]` |
| ho | `[]` | (thiếu entity) |
| đau nhức | `[]` | (thiếu entity) |
| sốt đau | `[]` | `[]` |
| táo bón (×2) | `[]` ×2 | `["isHistorical","isFamily"]` ×1 |
| lo âu (×2) | `[]` ×2 | `["isHistorical"]` ×1 |
| mất ngủ | `[]` | (thiếu entity) |

---

## 3. Code

| Thành phần | Trạng thái |
|-----------|-----------|
| Position | Không lỗi — xử lý đúng duplicate nếu LLM trả đủ |
| RxNorm linker | Hoạt động (cần verify version) |
| ICD-10 linker | Hoạt động (local KB từ anchor set) |
| Scorer | Đã có (`scorer.py`), hỗ trợ assertions array format |
| Main orchestration | LangChain LCEL chains + batch assertion, hoạt động ổn định |
| Ollama | llama3.1:8b (4.9GB), vừa RAM 16GB |

Các vấn đề A-D đều nằm ở tầng **prompt template**, không phải code.
