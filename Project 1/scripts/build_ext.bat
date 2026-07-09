@echo off
setlocal enabledelayedexpansion

REM This script activates the Conda environment and builds the CUDA extensions.
REM Since Conda might not be on the PATH directly via command prompt, we need to find it.

echo Initializing Conda...
call C:\Miniconda\Scripts\activate.bat 3dgs_vai
if errorlevel 1 (
    echo Failed to activate conda environment 3dgs_vai.
    exit /b 1
)

echo Conda environment activated: %CONDA_DEFAULT_ENV%

cd /d "%~dp0..\gaussian-splatting"
echo Building diff-gaussian-rasterization...
pip install -e submodules\diff-gaussian-rasterization

echo Building simple-knn...
pip install -e submodules\simple-knn

echo Done.
