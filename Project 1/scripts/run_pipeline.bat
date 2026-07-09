@echo off
setlocal enabledelayedexpansion

call conda activate 3dgs_vai
REM Setup paths
set PROJECT_DIR=%~dp0..
set DATA_DIR=%PROJECT_DIR%\VAI_NVS_DATA\phase1\public_set
set OUTPUT_DIR=%PROJECT_DIR%\output

echo Starting VAI NVS Pipeline...

REM List of scenes to run
set SCENES=hcm0031

for %%s in (%SCENES%) do (
    echo.
    echo ========================================================
    echo Processing Scene: %%s
    echo ========================================================
    
    set "SCENE_DIR=!DATA_DIR!\%%s"
    set "SCENE_OUT=!OUTPUT_DIR!\%%s"
    set "RENDER_OUT=!SCENE_OUT!\renders"
    
    REM 1. Train
    echo [1/3] Training 3DGS...
    python "!PROJECT_DIR!\scripts\train_scene.py" --scene_dir "!SCENE_DIR!" --model_path "!SCENE_OUT!" --config "!PROJECT_DIR!\configs\default.yaml"
    if errorlevel 1 (
        echo Training failed for %%s, skipping...
    ) else (
        REM 2. Render
        echo [2/3] Rendering test images...
        python "!PROJECT_DIR!\scripts\render_test.py" --scene_dir "!SCENE_DIR!" --model_path "!SCENE_OUT!" --output_dir "!RENDER_OUT!"
        if errorlevel 1 (
            echo Rendering failed for %%s, skipping evaluation...
        ) else (
            REM 3. Evaluate
            echo [3/3] Evaluating metrics...
            python "!PROJECT_DIR!\scripts\evaluate.py" --pred_dir "!RENDER_OUT!" --gt_dir "!SCENE_DIR!\test\images" --output_file "!SCENE_OUT!\metrics.json"
        )
    )

    echo Finished processing %%s.
)

echo.
echo All scenes processed. Creating submission zip...
python "%PROJECT_DIR%\scripts\make_submission.py" --output_dir "%OUTPUT_DIR%"

echo Done.
