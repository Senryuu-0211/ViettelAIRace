@echo off
setlocal enabledelayedexpansion

set CONDA_PATH=C:\Miniconda
set ENV_NAME=3dgs_vai
set PROJECT_DIR=D:\ViettelAIRace\Project 1
set DATA_DIR=%PROJECT_DIR%\VAI_NVS_DATA_ROUND2_UNDIST
set OUTPUT_DIR=%PROJECT_DIR%\output_round2_prod
set CONFIG=%PROJECT_DIR%\configs\production.yaml

title VAI NVS Production Pipeline

echo ================================================================
echo  PRODUCTION PIPELINE - 50K iters, undistorted, antialias
echo  Started: %date% %time%
echo ================================================================

for /d %%d in ("%DATA_DIR%\*") do (
    set "SCENE=%%~nxd"
    set "OUT=%OUTPUT_DIR%\!SCENE!"
    set "RENDER=%OUT%\renders"
    
    echo.
    echo ================================================================
    echo  [!date! !time!] TRAINING: !SCENE!
    echo ================================================================
    
    %CONDA_PATH%\Scripts\conda.exe run -n %ENV_NAME% python "%PROJECT_DIR%\scripts\train_scene.py" --scene_dir "%%d" --model_path "!OUT!" --config "%CONFIG%"
    if !ERRORLEVEL! NEQ 0 (
        echo ERROR: Training failed for !SCENE!
        pause
        exit /b 1
    )
    
    echo.
    echo [!date! !time!] RENDERING: !SCENE!
    mkdir "!RENDER!" 2>nul
    %CONDA_PATH%\Scripts\conda.exe run -n %ENV_NAME% python "%PROJECT_DIR%\scripts\render_test.py" --scene_dir "%%d" --model_path "!OUT!" --output_dir "!RENDER!"
    if !ERRORLEVEL! NEQ 0 (
        echo ERROR: Render failed for !SCENE!
        pause
        exit /b 1
    )
    
    echo [!date! !time!] DONE: !SCENE!
)

echo.
echo ================================================================
echo  All scenes done. Creating submission...
echo ================================================================
%CONDA_PATH%\python.exe python "%PROJECT_DIR%\scripts\make_submission.py" --output_dir "%OUTPUT_DIR%" --private_dir "%DATA_DIR%" --zip_name "submission_production.zip"

echo.
echo PIPELINE COMPLETE at %date% %time%
pause
