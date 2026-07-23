# AGENTS.md — Rules for AI Assistant

## NGUYÊN TẮC SỐ 1: KHÔNG TỰ Ý CHẠY LỆNH
- **LUÔN HỎI TRƯỚC** khi chạy bất kỳ lệnh nào làm thay đổi hệ thống (train, cài đặt, xóa file, git, docker, pip install, v.v.)
- Chỉ tự động chạy các lệnh **đọc** không gây hại: `ls`, `cat`, `git status`, `git diff`, kiểm tra file, đọc log
- Với mọi lệnh ghi/xóa/thực thi dài, phải nói rõ: **làm gì, ảnh hưởng gì, mất bao lâu**

## NGUYÊN TẮC SỐ 2: GIẢI THÍCH TRƯỚC KHI LÀM
- Trước mỗi hành động, nói ngắn gọn: việc này để làm gì, expected outcome là gì
- Nếu có nhiều cách, đưa ra options để user chọn

## NGUYÊN TẮC SỐ 3: KHÔNG CHẠY LỆNH DÀI TRONG BACKGROUND
- Không tự ý chạy train, build, hay process dài mà không hỏi
- Nếu cần chạy pipeline dài (>5 phút), thông báo thời gian dự kiến và hỏi xác nhận

## NGUYÊN TẮC SỐ 4: CLEAN & MINIMAL
- Không tạo file rác, script test, log thừa
- Dọn dẹp sau khi làm xong

## NGUYÊN TẮC SỐ 5: TIẾNG VIỆT
- Giao tiếp bằng tiếng Việt
