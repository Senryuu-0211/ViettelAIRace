@echo off
setlocal

set CUDA_HOME=C:\Miniconda\envs\scaffold_gs\Library
set CUDA_PATH=C:\Miniconda\envs\scaffold_gs\Library

echo === Building simple-knn ===
call conda run -n scaffold_gs pip install "%~dp0..\Scaffold-GS\submodules\simple-knn" --no-build-isolation
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: simple-knn
    exit /b 1
)

echo === Building diff-gaussian-rasterization ===
call conda run -n scaffold_gs pip install "%~dp0..\Scaffold-GS\submodules\diff-gaussian-rasterization" --no-build-isolation
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: diff-gaussian-rasterization
    exit /b 1
)

echo === All builds successful! ===
