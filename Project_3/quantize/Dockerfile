FROM vllm/vllm-openai:v0.22.1

# Download pre-quantized AWQ model từ HuggingFace
# Dùng huggingface_hub CLI có sẵn trong vLLM image
RUN pip install -q huggingface_hub && \
    huggingface-cli download QuantTrio/Qwen3.5-2B-AWQ --local-dir /quantized-model
