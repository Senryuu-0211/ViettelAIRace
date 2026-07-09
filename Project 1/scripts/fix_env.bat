@echo off
echo [1/5] Removing bad conda packages...
call conda remove -n 3dgs_vai pytorch torchvision torchaudio lpips tensorboard -y

echo [2/5] Cleaning pip cache for torch...
call conda run -n 3dgs_vai pip uninstall torch torchvision torchaudio lpips tensorboard -y

echo [3/5] Installing correct GPU PyTorch...
call conda run -n 3dgs_vai pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

echo [4/5] Installing lpips and tensorboard via pip...
call conda run -n 3dgs_vai pip install lpips tensorboard

echo [5/5] Building 3DGS submodules...
call conda run -n 3dgs_vai pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization
call conda run -n 3dgs_vai pip install -e gaussian-splatting/submodules/simple-knn
call conda run -n 3dgs_vai pip install -e gaussian-splatting/submodules/fused-ssim

echo All done! Testing PyTorch...
call conda run -n 3dgs_vai python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
