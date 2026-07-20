@echo off
echo === Step 1: Setting up MSVC ===
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

echo === Step 2: Setting CUDA 11.8 paths ===
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "CUDAFLAGS=-allow-unsupported-compiler"
set "TORCH_CUDA_ARCH_LIST=7.5;8.6"

echo === Step 3: Activating conda env ===
call conda activate 3dgs_vai

echo === Step 4: Installing PyTorch 2.3.1 CUDA 11.8 ===
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo FAILED: PyTorch install
    exit /b 1
)

echo === Step 5: Installing dependencies ===
pip install plyfile tqdm Pillow lpips ninja
if errorlevel 1 (
    echo FAILED: dependencies install
    exit /b 1
)

echo === Step 6: Verifying PyTorch CUDA ===
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

echo === Step 7: Building diff-gaussian-rasterization ===
cd gaussian-splatting\submodules\diff-gaussian-rasterization
python setup.py install
if errorlevel 1 (
    echo FAILED: diff-gaussian-rasterization
    cd ..\..\..
    exit /b 1
)

echo === Step 8: Building simple-knn ===
cd ..\simple-knn
python setup.py install
if errorlevel 1 (
    echo FAILED: simple-knn
    cd ..\..\..
    exit /b 1
)

echo === Step 9: Building fused-ssim ===
cd ..\fused-ssim
python setup.py install
if errorlevel 1 (
    echo FAILED: fused-ssim
    cd ..\..\..
    exit /b 1
)

cd ..\..\..
echo === ALL DONE! Environment ready! ===
