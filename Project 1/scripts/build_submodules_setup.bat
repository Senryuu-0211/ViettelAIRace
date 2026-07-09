@echo off
echo Setting up MSVC...
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"

echo Activating conda env...
call conda activate 3dgs_vai

echo Installing ninja...
pip install ninja

echo Building diff-gaussian-rasterization...
cd gaussian-splatting\submodules\diff-gaussian-rasterization
python setup.py install

echo Building simple-knn...
cd ..\simple-knn
python setup.py install

echo Building fused-ssim...
cd ..\fused-ssim
python setup.py install

cd ..\..\..
