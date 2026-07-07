# Medical Concept Extraction Pipeline Plan

## Mục tiêu

Xây dựng một pipeline sử dụng LLM để chuyển đổi văn bản y khoa tự do
thành danh sách các khái niệm y tế theo đúng schema của bài toán.

``` text
Clinical Note
        ↓
[
  {
    "text": "...",
    "type": "...",
    "assertions": [...],
    "candidates": [...],
    "position": [...]
  }
]
```

## Kiến trúc tổng thể

### 1. Self-hosted LLM (\~8B)

Sử dụng một mô hình LLM khoảng 8B tham số chạy cục bộ.

### 2. LangGraph

Sử dụng LangGraph để orchestration workflow:

-   Quản lý state
-   Điều phối các node
-   Fan-out / Fan-in
-   Retry nếu cần
-   Logging

Không sử dụng Agent theo kiểu ReAct, chỉ sử dụng LangGraph như một
workflow engine.

------------------------------------------------------------------------

## Agent State

``` python
class AgentState(TypedDict):
    # Original input
    input_text: str

    # Stage 1 outputs
    drugs: list
    diagnoses: list
    symptoms: list
    lab_names: list
    lab_results: list

    # Stage 2 outputs
    drugs_with_assertion: list
    diagnoses_with_assertion: list
    symptoms_with_assertion: list

    # Stage 3
    entities: list

    # Final Output
    final_output: list
```

Toàn bộ node đều đọc `input_text` từ state thay vì truyền lại giữa các
node.

------------------------------------------------------------------------

## Stage 1 -- Song song 5 Extractor

Input được đưa đồng thời vào 5 node độc lập.

``` text
                     Input
                       │
 ──────────────────────┼──────────────────────
 │          │          │          │          │
 ▼          ▼          ▼          ▼          ▼
Drug     Diagnosis  Symptom   Lab Name   Lab Result
Extractor Extractor Extractor Extractor  Extractor
```

Mỗi extractor có một prompt template riêng và chỉ chịu trách nhiệm cho
đúng một loại thực thể:

-   THUỐC
-   CHẨN_ĐOÁN
-   TRIỆU_CHỨNG
-   TÊN_XÉT_NGHIỆM
-   KẾT_QUẢ_XÉT_NGHIỆM

Output của mỗi extractor:

``` json
{
  "text": "...",
  "type": "..."
}
```

------------------------------------------------------------------------

## Stage 2 -- Assertion Model

Chỉ áp dụng cho:

-   THUỐC
-   CHẨN_ĐOÁN
-   TRIỆU_CHỨNG

Ba luồng này được đưa vào Assertion Model trước khi Merge.

``` text
Drug Extractor --------┐
                       │
Diagnosis Extractor ---┼────► Assertion Model
                       │
Symptom Extractor -----┘
```

Input:

-   input_text
-   entity
-   entity type

Output:

``` json
{
  "text": "...",
  "type": "...",
  "assertions": [
    "isNegated",
    "isFamily",
    "isHistorical"
  ]
}
```

------------------------------------------------------------------------

## Stage 3 -- Merge

Sau khi hoàn thành Assertion Model, toàn bộ entity được hợp nhất.

``` text
Drug (+assertion)

Diagnosis (+assertion)

Symptom (+assertion)

Lab Name

Lab Result

        ↓

      Merge
```

------------------------------------------------------------------------

## Stage 4 -- Candidate Mapping

Chỉ thực hiện với:

-   THUỐC
-   CHẨN_ĐOÁN

Pipeline:

``` text
Entity Name
      ↓
API / Database
      ↓
Candidate Codes
```

-   THUỐC → RxNorm
-   CHẨN_ĐOÁN → ICD-10

------------------------------------------------------------------------

## Stage 5 -- Position

Không sử dụng LLM.

Tính vị trí bằng Python thông qua việc tìm vị trí xuất hiện của chuỗi
trong văn bản.

------------------------------------------------------------------------

## Stage 6 -- Final JSON

Ghép toàn bộ kết quả thành đúng schema JSON theo yêu cầu của bài toán.
