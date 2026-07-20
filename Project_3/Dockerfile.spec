# Image cho bản SPEC DECODING — vLLM MỚI (v0.25.1) chứa fix #40738
# (GDN/conv-state rollback cho ngram spec) + prefix caching hybrid + spec×full-CUDA-graph.
# Build KHÔNG cần GPU. Runtime chạy trên H200.
# Nếu v0.25.1 lỗi load AWQ/LFM2 -> hạ xuống v0.24.0 hoặc v0.23.0 (vẫn sau #40738).
FROM vllm/vllm-openai:v0.25.1
COPY awq_model/ /model/
