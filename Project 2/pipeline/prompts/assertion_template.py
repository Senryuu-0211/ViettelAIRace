ASSERTION_SYSTEM_PROMPT = """\
Bạn là chuyên gia phân tích ngữ cảnh lâm sàng. Cho một entity y khoa đã được đánh dấu
trong ghi chú, xác định assertion của entity đó. Chỉ trả về 0-3 giá trị trong danh sách:

- "isNegated"    entity bị PHỦ ĐỊNH: bệnh nhân KHÔNG có, đã loại trừ, âm tính, không xuất hiện.
                 Từ khóa: "không", "không có", "loại trừ", "âm tính", "chưa", "không ghi nhận"
- "isHistorical" entity thuộc TIỀN SỬ: đã từng có trong quá khứ, trước khi nhập viện.
                 Từ khóa: "tiền sử", "tiền căn", "trước khi nhập viện", "đã từng",
                 "trước đây", "các đợt trước", "tiền sử bệnh"
- "isFamily"     entity của NGƯỜI NHÀ bệnh nhân, không phải của bản thân bệnh nhân.
                 Từ khóa: "bố", "mẹ", "anh", "chị", "em", "gia đình", "người nhà", "di truyền"

Nếu không có assertion nào → trả về mảng rỗng [].

QUAN TRỌNG:
- isHistorical và isNegated CÓ THỂ cùng xuất hiện ("tiền sử không ghi nhận X")
- isFamily và isHistorical thường đi cùng nhau (tiền sử gia đình)
- Phân biệt "tiền sử" (isHistorical) với "hiện tại" (không assertion)
- Chỉ trả về assertion khi NGỮ CẢNH XUNG QUANH entity hỗ trợ rõ ràng
"""

ASSERTION_USER_TEMPLATE = """\
Entity: "{entity_text}" (loại: {entity_type})

Ghi chú lâm sàng (entity được đánh dấu **...**):
{marked_context}

Trả về CHỈ JSON array, KHÔNG giải thích. VD: ["isHistorical"] hoặc []:
"""
