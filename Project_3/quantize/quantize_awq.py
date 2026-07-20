"""
AWQ INT4 Quantization cho Qwen3.5-2B
=====================================
Yêu cầu:
  pip install autoawq transformers torch
  GPU NVIDIA >= 8GB VRAM

Sử dụng:
  cd d:\ViettelAIRace\Project_3\quantize
  python quantize_awq.py
"""

import json
import os
import sys

def main():
    # Lazy imports to give better error messages
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: Cần cài đặt autoawq và transformers:")
        print("  pip install autoawq transformers")
        sys.exit(1)

    MODEL_ID = "Qwen/Qwen3.5-2B"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "qwen3.5-2b-awq")
    TRACE_FILE = os.path.join(os.path.dirname(__file__), "..", "trace-round1.jsonl")

    print("=" * 60)
    print("AWQ INT4 Quantization - Qwen3.5-2B")
    print("=" * 60)

    # --- Bước 1: Chuẩn bị calibration data từ trace ---
    print("\n[1/4] Chuẩn bị calibration data từ trace...")
    calib_data = []
    with open(TRACE_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20:  # 20 conversation đầu tiên (mỗi cái là unique)
                break
            req = json.loads(line)
            messages = req["body"]["messages"]
            # Nối toàn bộ nội dung conversation làm calibration
            text = "\n".join(m["content"] for m in messages)
            calib_data.append(text)

    print(f"  Số mẫu calibration: {len(calib_data)}")
    print(f"  Độ dài trung bình: {sum(len(t) for t in calib_data) // len(calib_data)} chars")

    # --- Bước 2: Load tokenizer ---
    print("\n[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # --- Bước 3: Load model và quantize ---
    print("\n[3/4] Loading model và quantizing (có thể mất 10-30 phút)...")
    model = AutoAWQForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        safetensors=True,
    )

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",  # vLLM sẽ tự chọn Marlin kernel khi load
    }

    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data,
    )

    # --- Bước 4: Save ---
    print(f"\n[4/4] Saving quantized model tới {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_quantized(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Tính tổng dung lượng
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))

    print("\n" + "=" * 60)
    print(f"DONE! Model đã quantize tại: {OUTPUT_DIR}")
    print(f"Dung lượng: {total_size / 1024 / 1024:.1f} MB")
    print(f"\nBước tiếp theo:")
    print(f"  1. cd {os.path.dirname(__file__)}")
    print(f"  2. docker build -t <YOUR_DOCKERHUB>/vllm-qwen-awq:latest .")
    print(f"  3. docker push <YOUR_DOCKERHUB>/vllm-qwen-awq:latest")
    print("=" * 60)


if __name__ == "__main__":
    main()
