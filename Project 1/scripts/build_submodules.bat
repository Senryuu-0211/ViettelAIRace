@echo off
call conda activate 3dgs_vai
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization --no-build-isolation
pip install -e gaussian-splatting/submodules/simple-knn --no-build-isolation
pip install -e gaussian-splatting/submodules/fused-ssim --no-build-isolation
