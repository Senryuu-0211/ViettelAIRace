@echo off
setlocal enabledelayedexpansion

set CONDA_ENV=scaffold_gs
set PROJECT_DIR=%~dp0..
set OUTPUT_DIR=%PROJECT_DIR%\output_scaffold
set SCAFFOLD_DIR=%PROJECT_DIR%\Scaffold-GS
set CONFIG_FILE=%PROJECT_DIR%\configs\scaffold.yaml

:: VS env setup for CUDA
set "CUDA_HOME=C:\Miniconda\envs\scaffold_gs\Library"
set "DISTUTILS_USE_SDK=1"

:: Parse arguments
set "MODE=default"
:ParseArgs
if "%~1"=="" goto :DoneArgs
if /i "%~1"=="--mode" (
    set "MODE=%~2"
    shift
    shift
    goto :ParseArgs
)
shift
goto :ParseArgs
:DoneArgs

set ROUND2_DIR=%PROJECT_DIR%\VAI_NVS_DATA_ROUND2

echo ================================================================
echo  VAI NVS Pipeline - Scaffold-GS [Mode: %MODE%]
echo  Config: %CONFIG_FILE%
echo ================================================================

for /d %%d in ("%ROUND2_DIR%\*") do (
    set "SCENE_NAME=%%~nxd"
    call :ProcessScene "%%d" "!SCENE_NAME!"
)

goto :Finish

:ProcessScene
set "SCENE_DIR=%~1"
set "SCENE_NAME=%~2"
set "SCENE_OUT=%OUTPUT_DIR%\%SCENE_NAME%"
set "RENDER_OUT=%SCENE_OUT%\renders"

echo.
echo ========================================================
echo Processing Scene: %SCENE_NAME%
echo ========================================================

if exist "%SCENE_OUT%\metrics.json" (
    echo Scene %SCENE_NAME% already fully processed. Skipping...
    goto :eof
)

:: 1. Train
echo [1/2] Training Scaffold-GS...
set SOURCE_PATH=%SCENE_DIR%\train

call conda run -n %CONDA_ENV% python "%SCAFFOLD_DIR%\train.py" ^
    -s "%SOURCE_PATH%" ^
    -m "%SCENE_OUT%" ^
    --iterations 7000 ^
    --feat_dim 32 ^
    --n_offsets 10 ^
    --voxel_size 0.001 ^
    --lambda_dssim 0.2 ^
    --feature_lr 0.0075 ^
    --opacity_lr 0.02 ^
    --scaling_lr 0.007 ^
    --rotation_lr 0.002 ^
    --offset_lr_init 0.01 ^
    --offset_lr_final 0.0001 ^
    --start_stat 500 ^
    --update_from 1500 ^
    --update_until 5000 ^
    --update_interval 100 ^
    --gpu 0

if %ERRORLEVEL% NEQ 0 (
    echo Training failed for %SCENE_NAME%, skipping...
    goto :eof
)

:: 2. Render
echo [2/2] Rendering test images...
call conda run -n %CONDA_ENV% python "%SCAFFOLD_DIR%\render.py" ^
    -m "%SCENE_OUT%" ^
    --skip_train

if %ERRORLEVEL% NEQ 0 (
    echo Rendering failed for %SCENE_NAME%, skipping...
    goto :eof
)

:: Mark as done
echo {"psnr": 0.0} > "%SCENE_OUT%\metrics.json"

echo Finished processing %SCENE_NAME%.
goto :eof

:Finish
echo.
echo All scenes processed. Creating submission zip...
python "%PROJECT_DIR%\scripts\make_submission.py" --output_dir "%OUTPUT_DIR%" --private_dir "%ROUND2_DIR%" --zip_name "submission_scaffold.zip"

echo Done.
