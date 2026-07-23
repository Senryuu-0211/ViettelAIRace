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


CONV_ALL = r"re:^model\.layers\.\d+\.conv\."
CONV_OUT = r"re:^model\.layers\.\d+\.conv\.out_proj$"


def build_recipe(stage: str):
    """
    stage="lmhead" : conv VẪN bỏ qua (như checkpoint cũ) + nén lm_head.
                     -> CHẠY ĐƯỢC VỚI vLLM GỐC, không cần vá. Cắt 198 MB.
    stage="conv"   : nén conv proj, KHÔNG nén lm_head.
                     -> BẮT BUỘC có bản vá tools/patch_shortconv.py. Cắt 248 MB.
    stage="full"   : nén cả hai. -> BẮT BUỘC có bản vá. Cắt 446 MB.
    """
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    quantize_conv = stage in ("conv", "full")
    quantize_head = stage in ("lmhead", "full")

    # --- Nhóm 1: AWQ W4A16 cho các layer có norm đứng trước (FFN, attention,
    #     và conv.in_proj nếu được bật) ---
    awq_ignore = ["lm_head"]
    awq_ignore.append(CONV_OUT if quantize_conv else CONV_ALL)
    awq = AWQModifier(targets=["Linear"], ignore=awq_ignore, scheme=SCHEME)

    # --- Nhóm 2: RTN W4A16 cho những layer không có norm đứng trước ---
    rtn_targets = []
    if quantize_conv:
        rtn_targets.append(CONV_OUT)
    if quantize_head:
        # lm_head bị ĐỌC 268 MB MỖI step -> nén là +~2.4 điểm.
        # Cũng là chỗ rủi ro logit cao nhất -> luôn đo lại GPQA sau bước này.
        rtn_targets.append("lm_head")

    recipe = [awq]
    if rtn_targets:
        recipe.append(QuantizationModifier(targets=rtn_targets, scheme=SCHEME, ignore=[]))
    return recipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["lmhead", "conv", "full"], default="lmhead",
                    help="lmhead = chỉ nén lm_head (KHÔNG cần vá vLLM) | "
                         "conv = nén conv proj (CẦN vá) | full = cả hai (CẦN vá)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="chỉ nạp lại checkpoint đã có và in kích thước, không quantize")
    ap.add_argument("--calib-samples", type=int, default=512,
                    help="512 cho GPU >=24GB; dùng 256 trên RTX 3060 12GB")
    ap.add_argument("--seqlen", type=int, default=4096,
                    help="4096 khớp prompt thật trong trace; hạ 2048 nếu thiếu VRAM")
    ap.add_argument("--no-streaming", action="store_true",
                    help="tải TRỌN split calibration (~2-3 GB đĩa). Mặc định dùng "
                         "streaming: chỉ kéo đúng số mẫu cần, tốn ~50 MB")
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
    # Mặc định STREAMING -> chỉ kéo đúng args.calib_samples mẫu (~50 MB) thay vì
    # tải trọn split train_sft (~2-3 GB). Quan trọng khi ổ đĩa chật.
    if args.no_streaming:
        ds = (
            load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
            .shuffle(seed=42)
            .select(range(args.calib_samples))
        )
    else:
        import itertools
        from datasets import Dataset

        stream = load_dataset(
            "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True,
        ).shuffle(seed=42, buffer_size=2000)
        ds = Dataset.from_list(list(itertools.islice(stream, args.calib_samples)))

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
    # Mốc kích thước theo từng stage. Gốc 1.324 GB; lm_head 268->70 MB, conv 336->88 MB.
    BANDS = {
        "lmhead": (1.10, 1.16, "chỉ lm_head được nén, conv CỐ Ý giữ BF16"),
        "conv":   (1.05, 1.11, "chỉ conv được nén, lm_head giữ BF16"),
        "full":   (0.85, 0.91, "nén cả lm_head lẫn conv"),
    }
    lo, hi, note = BANDS[args.stage]
    gb = total / 1e9
    print(f"\n[xong] {out_dir} = {gb:.3f} GB")
    print(f"  mốc cho --stage {args.stage}: {lo:.2f}-{hi:.2f} GB  ({note})")
    if gb > hi:
        print(f"  ⚠️ LỚN HƠN MỐC -> recipe chưa nén được thứ cần nén cho stage này.")
        print(f"     Kiểm `ignore` trong {out_dir}/config.json.")
    elif gb < lo:
        print(f"  ⚠️ NHỎ HƠN MỐC -> có thể đã nén nhầm layer không định nén.")

    # Kiểm ĐỐI XỨNG ngay tại đây — đây là chỗ bản AWQ cũ hỏng, mất ~0.85 ms/token
    # (~6.5 điểm) vì có weight_zero_point nên trượt kernel Marlin.
    try:
        import json
        qc = json.load(open(os.path.join(out_dir, "config.json")))["quantization_config"]
        for gname, g in qc["config_groups"].items():
            w = g["weights"]
            ok = w.get("symmetric") is True
            print(f"  {gname}: num_bits={w['num_bits']} symmetric={w.get('symmetric')}"
                  f" {'✅' if ok else '❌ BẤT ĐỐI XỨNG -> requant lại với SCHEME=W4A16'}")
    except Exception as e:                       # noqa: BLE001
        print(f"  (không đọc được config.json để kiểm đối xứng: {e})")

    print("  BƯỚC TIẾP (tools/RUNBOOK_requant.md): nạp thử + xác nhận vào Marlin,")
    print("  rồi đo GPQA (cần >= 0.30 để giữ f_delta = 1).")


if __name__ == "__main__":
    main()
