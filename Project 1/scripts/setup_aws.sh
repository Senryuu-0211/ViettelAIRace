#!/usr/bin/env bash
# One-time setup for Scaffold-GS on g5.xlarge (Amazon Linux 2023 + A10G 24GB)
# Usage: bash ~/project/scripts/setup_aws.sh
set -euo pipefail

PROJECT_DIR="$HOME/project"
ENV_NAME="scaffold_gs"

echo "=== [1/5] Miniconda ==="
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$HOME/miniconda3"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "=== [2/5] Conda env: python 3.9 + CUDA 12.4 + torch 2.6.0 ==="
if ! conda env list | grep -q "${ENV_NAME}\b"; then
    conda create -n "$ENV_NAME" python=3.9 -y
fi
conda activate "$ENV_NAME"

# nvcc + CUDA headers via conda (no sudo, self-contained in env)
conda install -c nvidia/label/cuda-12.4.0 cuda-nvcc cuda-cudart-dev -y

# PyTorch (matches local working env)
pip install --no-input torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124

echo "=== [3/5] Python deps ==="
pip install --no-input plyfile tqdm einops lpips opencv-python-headless pyyaml laspy

echo "=== [4/5] Build CUDA extensions (A10G = sm_86) ==="
export TORCH_CUDA_ARCH_LIST="8.6"
cd "$PROJECT_DIR/Scaffold-GS"
pip install --no-input submodules/diff-gaussian-rasterization submodules/simple-knn

echo "=== [5/5] Verify ==="
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "FAIL: CUDA not available"
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name} ({gpu.total_memory // 1024**3} GB) | sm_{gpu.major}{gpu.minor}")
import diff_gaussian_rasterization
import simple_knn
print("CUDA extensions: OK")
EOF

echo ""
echo "=== ALL DONE ==="
echo "  tmux new -s train"
echo "  conda activate ${ENV_NAME} && cd ${PROJECT_DIR}"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/resume_scaffold.py --all 2>&1 | tee train.log"
