@echo off
setlocal enabledelayedexpansion

call conda activate 3dgs_vai

REM Setup paths BEFORE shift messes up %0
set PROJECT_DIR=%~dp0..
set OUTPUT_DIR=%PROJECT_DIR%\output_round2

REM Parse arguments
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

set CONFIG_FILE=%PROJECT_DIR%\configs\%MODE%.yaml

if not exist "%CONFIG_FILE%" (
    echo ERROR: Config file not found: %CONFIG_FILE%
    exit /b 1
)

set ROUND2_DIR=%PROJECT_DIR%\VAI_NVS_DATA_ROUND2

echo ================================================================
echo  VAI NVS Pipeline - Round 2 [Mode: %MODE%]
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
echo Processing Scene: %SCENE_NAME% [Mode: %MODE%]
echo ========================================================

if exist "%SCENE_OUT%\metrics.json" (
    echo Scene %SCENE_NAME% already fully processed. Skipping...
    goto :eof
)

REM 1. Train
echo [1/3] Training 3DGS...
python "%PROJECT_DIR%\scripts\train_scene.py" --scene_dir "%SCENE_DIR%" --model_path "%SCENE_OUT%" --config "%CONFIG_FILE%"
if errorlevel 1 (
    echo Training failed for %SCENE_NAME%, skipping...
    goto :eof
)

REM 2. Render
echo [2/3] Rendering test images...
python "%PROJECT_DIR%\scripts\render_test.py" --scene_dir "%SCENE_DIR%" --model_path "%SCENE_OUT%" --output_dir "%RENDER_OUT%"
if errorlevel 1 (
    echo Rendering failed for %SCENE_NAME%, skipping...
    goto :eof
)

REM 3. Evaluate (Dummy metrics since no GT for hidden test set)
echo [3/3] Creating dummy metrics to mark as done...
echo {"psnr": 0.0} > "%SCENE_OUT%\metrics.json"

echo Finished processing %SCENE_NAME%.
goto :eof

:Finish
echo.
echo All scenes processed. Creating submission zip...
python "%PROJECT_DIR%\scripts\make_submission.py" --output_dir "%OUTPUT_DIR%" --private_dir "%ROUND2_DIR%" --zip_name "submission_round2.zip"

echo Done.
