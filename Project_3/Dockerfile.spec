# Image cho bản SPEC DECODING — vLLM v0.25.1 chứa fix #40738
# (conv/SSM-state rollback cho ngram spec trên hybrid model).
# Build KHÔNG cần GPU. Runtime chạy trên H200.
# Các tag tồn tại: v0.25.1-ubuntu2404, v0.25.1-cu129-ubuntu2404, v0.25.1
# Nếu tag này lỗi load AWQ/LFM2 -> thử v0.25.1-ubuntu2404 hoặc nightly.
FROM vllm/vllm-openai:v0.25.1-cu129-ubuntu2404
COPY awq_model/ /model/
