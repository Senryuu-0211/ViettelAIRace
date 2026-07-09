@echo off
echo Setting up MSVC Community...
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

echo Activating conda env...
call conda activate 3dgs_vai

echo Checking cl.exe...
where cl.exe
cl.exe

echo Checking nvcc...
where nvcc
nvcc --version

echo Uninstalling ninja to force distutils (verbose error)...
pip uninstall ninja -y

echo Building diff-gaussian-rasterization...
cd gaussian-splatting\submodules\diff-gaussian-rasterization
python setup.py install
