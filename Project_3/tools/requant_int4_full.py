"""
Requantize LFM2.5-1.2B sang INT4 ĐẦY ĐỦ — nén nốt phần checkpoint cũ bỏ sót.

BỐI CẢNH — SỐ ĐO THẬT từ header của awq_model/model.safetensors (1323.7 MB):

    | Nhóm                        | dtype       |     MB | Ghi chú                       |
    |-----------------------------|-------------|--------|-------------------------------|
    | lm_head + embed_tokens      | BF16        |  536.9 | CHƯA NÉN (2 x 268.4, xem dưới)|
    | FFN (w1/w2/w3)              | I32 4-bit   |  418.4 | đã nén                        |
    | conv.in_proj / conv.out_proj| BF16        |  335.7 | CHƯA NÉN (10 layer conv)      |
    | attention (q/k/v/out)       | I32 4-bit   |   32.7 | đã nén                        |

    => 886 MB (66.9%) VẪN Ở DẠNG FLOAT. Bản này chỉ nén được 1/3.

    Byte thực sự phải ĐỌC mỗi step decode = 1323.7 - 268.4 (embed_tokens là gather,
    không phải GEMM) = ~1055 MB.  Ở MiG 1g.18gb (~600 GB/s) => ~1.76 ms/token.
    Đo được TPOT 4 ms  =>  ~2.1 ms còn lại là overhead host (launch/scheduler/SSE).

    Sau khi nén nốt:  conv 335.7 -> ~88 MB,  lm_head 268.4 -> ~70 MB
                      => đọc ~609 MB/step => ~1.02 ms  (tiết kiệm 0.74 ms)
    Cộng với cudagraph FULL cắt overhead 2.1 -> ~0.4 ms:
                      TPOT ~= 1.02 + 0.16 (KV) + 0.4 = ~1.58 ms  =>  ~83 diem.

    Đó là lý do bản AWQ cũ ra điểm y hệt FP8: cả hai đều chưa cắt được bao nhiêu byte.

⚠️ NGHI VẤN KERNEL — có thể là nguyên nhân thật sự:
    Checkpoint hiện tại có `weight_zero_point` (I32) => lượng tử hoá BẤT ĐỐI XỨNG
    (scheme W4A16_ASYM). Đường nhanh Marlin của vLLM ưu tiên dạng ĐỐI XỨNG; asym dễ
    rơi xuống kernel chậm hơn, ăn hết phần lợi băng thông. Vì vậy script này mặc định
    dùng "W4A16" (đối xứng). Nếu độ chính xác tụt quá thì mới quay lại ASYM.

NGÂN SÁCH ĐỘ CHÍNH XÁC ĐANG BỎ PHÍ:
    Luật cho f(delta)=1 khi accuracy_drop <= 0.10. Bản hiện tại đo được drop = 0.
    Tức ta đang KHÔNG tiêu gì trong ngân sách 0.10 -> nén mạnh tay là "miễn phí"
    miễn là còn dưới ngưỡng. PHẢI đo lại GPQA sau khi requant để xác nhận.

CHẠY Ở ĐÂU: bất kỳ GPU >= 12 GB. KHÔNG cần H200 — đây là bước hiệu chuẩn offline,
chỉ tốn thời gian chứ không cần đúng phần cứng chấm. RTX 3060 12GB chạy được với
--calib-samples 256 --seqlen 2048 (khoảng 1-2 tiếng).

    pip install llmcompressor==0.13.* datasets transformers accelerate
    python requant_int4_full.py --stage conv --calib-samples 256 --seqlen 2048
    python requant_int4_full.py --stage full --calib-samples 256 --seqlen 2048

RỦI RO ĐÃ BIẾT — đọc trước khi chạy:
  1. vLLM có thể CHƯA hỗ trợ nạp conv.in_proj/out_proj đã nén cho kiến trúc Lfm2.
     -> BẮT BUỘC smoke-test nạp model trước khi nộp (xem --verify).
  2. lm_head INT4 làm hỏng logit mạnh hơn các layer khác -> làm ở stage riêng,
     đo GPQA riêng, sẵn sàng bỏ nếu drop > 0.10.
  3. AWQ smoothing cần một norm đứng trước layer. conv.in_proj có operator_norm nên
     hợp lệ; conv.out_proj thì KHÔNG -> dùng RTN (QuantizationModifier) cho nhóm này
     thay vì AWQ. Đó là lý do script chia 2 modifier.
"""

import argparse
import os

MODEL_SRC = os.environ.get("MODEL_SRC", "LiquidAI/LFM2.5-1.2B-Instruct")

# Đối xứng để có cửa dùng kernel Marlin. Đổi sang "W4A16_ASYM" nếu accuracy tụt quá.
SCHEME = os.environ.get("SCHEME", "W4A16")


def build_recipe(stage: str):
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    # --- Nhóm 1: FFN + attention + conv.in_proj -> AWQ W4A16 (có norm đứng trước) ---
    awq = AWQModifier(
        targets=["Linear"],
        # KHÁC recipe cũ: KHÔNG còn ignore 're:^model\\.layers\\.\\d+\\.conv\\.'
        ignore=["lm_head", r"re:^model\.layers\.\d+\.conv\.out_proj$"],
        scheme=SCHEME,
    )

    # --- Nhóm 2: conv.out_proj -> RTN W4A16 (không cần smoothing mapping) ---
    rtn_targets = [r"re:^model\.layers\.\d+\.conv\.out_proj$"]
    if stage == "full":
        # lm_head đọc 268 MB MỖI step -> nén được là +~2.5 điểm. Rủi ro logit cao nhất.
        rtn_targets.append("lm_head")

    rtn = QuantizationModifier(
        targets=rtn_targets,
        scheme=SCHEME,
        ignore=[],
    )
    return [awq, rtn]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["conv", "full"], default="conv",
                    help="conv = nén thêm conv proj | full = nén thêm cả lm_head")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="chỉ nạp lại checkpoint đã có và in kích thước, không quantize")
    ap.add_argument("--calib-samples", type=int, default=512,
                    help="512 cho GPU >=24GB; dùng 256 trên RTX 3060 12GB")
    ap.add_argument("--seqlen", type=int, default=4096,
                    help="4096 khớp prompt thật trong trace; hạ 2048 nếu thiếu VRAM")
    args = ap.parse_args()

    out_dir = args.out or f"awq_model_int4_{args.stage}"

    if args.verify:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True)
        nbytes = sum(p.numel() * p.element_size() for p in m.parameters())
        print(f"[verify] {out_dir}: {nbytes/1e9:.3f} GB tham số thường trú")
        return

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot

    tok = AutoTokenizer.from_pretrained(MODEL_SRC)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_SRC, torch_dtype="auto", device_map="cuda:0", trust_remote_code=True,
    )

    # Calibration: prompt dài ~4K token cho khớp phân phối thật của trace.
    ds = (
        load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        .shuffle(seed=42)
        .select(range(args.calib_samples))
    )

    def _fmt(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        return tok(text, truncation=True, max_length=args.seqlen, add_special_tokens=False)

    ds = ds.map(_fmt, remove_columns=ds.column_names)

    oneshot(
        model=model,
        dataset=ds,
        recipe=build_recipe(args.stage),
        max_seq_length=args.seqlen,
        num_calibration_samples=args.calib_samples,
        output_dir=out_dir,
    )

    total = 0
    for root, _, files in os.walk(out_dir):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    target = "1.05-1.10 GB" if args.stage == "conv" else "0.85-0.90 GB"
    print(f"\n[xong] {out_dir} = {total/1e9:.3f} GB   (mốc cần đạt: {target})")
    if total / 1e9 > 1.25:
        print("  ⚠️ VẪN ~1.3 GB -> recipe KHÔNG ăn vào conv proj. Xem CỬA 0 trong")
        print("     tools/RUNBOOK_requant.md trước khi làm tiếp.")
    print("  BƯỚC TIẾP (tools/RUNBOOK_requant.md): CỬA 3 smoke-test vLLM nạp được +")
    print("  đúng kernel, CỬA 4 đo lại GPQA (cần >= 0.30 để giữ f_delta = 1).")


if __name__ == "__main__":
    main()
