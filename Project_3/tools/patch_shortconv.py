"""
Vá vLLM để layer conv của LFM2 nhận được quant_config.

VÌ SAO: `ShortConv.__init__` không có tham số `quant_config`, nên hai lớp
`in_proj`/`out_proj` được tạo với `UnquantizedLinearMethod` -> compressed-tensors
bỏ qua chúng -> 335.7 MB (10 layer conv) mãi mãi ở BF16, dù checkpoint nén kiểu gì.
vLLM vốn có đủ cơ chế lượng tử hoá; chỉ là upstream quên nối dây tới layer này.

BẢN VÁ KHÔNG ĐỔI MỘT PHÉP TÍNH NÀO. Chỉ thêm 3 chỗ:
    1. ShortConv.__init__            thêm tham số  quant_config=None
    2. self.in_proj  = ...(...)      thêm  quant_config=quant_config
    3. self.out_proj = ...(...)      thêm  quant_config=quant_config
    4. lfm2.py: ShortConv(...)       thêm  quant_config=quant_config

DÙNG SCRIPT thay vì sửa tay để BTC dựng lại được, và để mọi sai lệch BÁO LỖI NGAY
thay vì hỏng âm thầm lúc chấm.

    python3 patch_shortconv.py --vllm-dir /usr/local/lib/python3.12/dist-packages/vllm
    python3 patch_shortconv.py --vllm-dir ... --check    # chỉ kiểm tra, không sửa
"""

import argparse
import difflib
import glob
import os
import re
import sys


class PatchError(RuntimeError):
    pass


def _find(models_dir, name):
    hits = glob.glob(os.path.join(models_dir, "**", name), recursive=True)
    if not hits:
        raise PatchError(f"không tìm thấy {name} dưới {models_dir}")
    if len(hits) > 1:
        raise PatchError(f"tìm thấy nhiều {name}: {hits} — chỉ định tay")
    return hits[0]


def patch_short_conv(src: str) -> str:
    """3 sửa đổi trong short_conv.py."""
    # --- (1) thêm quant_config vào chữ ký ShortConv.__init__ ---
    m = re.search(r"class\s+ShortConv\b", src)
    if not m:
        raise PatchError("không thấy `class ShortConv`")
    init = re.search(r"def\s+__init__\s*\((.*?)\)\s*(->[^:]*)?:", src[m.start():], re.S)
    if not init:
        raise PatchError("không thấy `def __init__` trong ShortConv")
    params = init.group(1)
    if "quant_config" in params:
        print("  [bỏ qua] __init__ đã có quant_config")
    else:
        abs_a = m.start() + init.start(1)
        abs_b = m.start() + init.end(1)
        new_params = params.rstrip()
        # giữ nguyên dấu phẩy cuối nếu có, để không phá style multi-line
        sep = "" if new_params.endswith(",") else ","
        new_params = f"{new_params}{sep} quant_config=None"
        src = src[:abs_a] + new_params + src[abs_b:]

    # --- (2)(3) truyền xuống hai Linear ---
    for which in ("in_proj", "out_proj"):
        anchor = re.compile(
            r'(prefix\s*=\s*f?["\'][^"\']*\.' + which + r'["\']\s*,?)'
        )
        hits = list(anchor.finditer(src))
        if len(hits) != 1:
            raise PatchError(
                f"kỳ vọng đúng 1 chỗ `prefix=...{which}`, thấy {len(hits)}. "
                f"Nguồn đã đổi — phải xem lại bằng tay."
            )
        h = hits[0]
        tail = src[h.end():h.end() + 200]
        if "quant_config" in tail.split(")")[0]:
            print(f"  [bỏ qua] {which} đã có quant_config")
            continue
        ins = h.group(1)
        ins = ins if ins.rstrip().endswith(",") else ins + ","
        src = src[:h.start()] + ins + " quant_config=quant_config," + src[h.end():]
    return src


def patch_lfm2(src: str) -> str:
    """Truyền quant_config vào chỗ gọi ShortConv(...)."""
    hits = list(re.finditer(r"ShortConv\s*\(", src))
    if len(hits) != 1:
        raise PatchError(
            f"kỳ vọng đúng 1 chỗ gọi `ShortConv(`, thấy {len(hits)} — xem lại bằng tay"
        )
    h = hits[0]
    # tìm dấu ) đóng của lời gọi
    depth, i = 0, h.end() - 1
    while i < len(src):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    else:
        raise PatchError("không tìm được dấu đóng của lời gọi ShortConv(")

    call = src[h.end():i]
    if "quant_config" in call:
        print("  [bỏ qua] lời gọi ShortConv đã truyền quant_config")
        return src

    # quant_config phải có trong scope — cảnh báo nếu không thấy trong hàm bao ngoài
    before = src[max(0, h.start() - 4000):h.start()]
    if "quant_config" not in before:
        print("  ⚠️  KHÔNG thấy `quant_config` trong phạm vi bao quanh lời gọi ShortConv.")
        print("      Có thể phải tự lấy nó từ vllm_config/quant_config ở hàm bao ngoài.")

    sep = "" if call.rstrip().endswith(",") or not call.strip() else ","
    return src[:i] + f"{sep} quant_config=quant_config" + src[i:]


def run(vllm_dir, check_only):
    models_dir = os.path.join(vllm_dir, "model_executor")
    targets = [
        (_find(models_dir, "short_conv.py"), patch_short_conv),
        (_find(models_dir, "lfm2.py"), patch_lfm2),
    ]

    changed = False
    for path, fn in targets:
        print(f"\n=== {path}")
        with open(path, encoding="utf-8") as f:
            old = f.read()
        new = fn(old)
        if old == new:
            print("  không có gì để đổi")
            continue
        changed = True
        diff = difflib.unified_diff(
            old.splitlines(True), new.splitlines(True),
            fromfile="gốc", tofile="đã vá", n=2,
        )
        sys.stdout.writelines(diff)
        if not check_only:
            import ast
            ast.parse(new)          # sai cú pháp -> chết ngay tại đây
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
            print("  -> đã ghi")

    if check_only:
        print("\n[--check] không ghi gì. Xem diff ở trên rồi chạy lại bỏ --check.")
    elif changed:
        print("\n[xong] Đã vá. BƯỚC TIẾP: nạp thử checkpoint đã requant rồi kiểm log —")
        print("       conv proj KHÔNG được nằm trong danh sách ignored nữa.")
    else:
        print("\n[xong] Không có thay đổi nào (có thể đã vá từ trước).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-dir", required=True,
                    help="vd /usr/local/lib/python3.12/dist-packages/vllm")
    ap.add_argument("--check", action="store_true", help="chỉ in diff, không ghi")
    args = ap.parse_args()
    try:
        run(args.vllm_dir, args.check)
    except PatchError as e:
        print(f"\n❌ VÁ THẤT BẠI: {e}", file=sys.stderr)
        print("   Nguồn vLLM khác với giả định. Mở file ra sửa tay theo 4 điểm "
              "mô tả ở đầu script này.", file=sys.stderr)
        sys.exit(1)
