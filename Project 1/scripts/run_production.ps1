$condaPath = "C:\Miniconda"
$envName = "3dgs_vai"
$projectDir = "D:\ViettelAIRace\Project 1"
$dataDir = "VAI_NVS_DATA_ROUND2_UNDIST"
$outputDir = "output_round2_prod"
$configFile = "configs\production.yaml"

$scenes = Get-ChildItem -Path "$projectDir\$dataDir" -Directory | ForEach-Object { $_.Name }

Write-Host "Starting production pipeline for $($scenes.Count) scenes..."
Write-Host "Scenes: $($scenes -join ', ')"

foreach ($scene in $scenes) {
    Write-Host ""
    Write-Host "=" * 60
    Write-Host "TRAINING: $scene (50K iterations)"
    Write-Host "=" * 60
    
    $sceneOutput = "$projectDir\$outputDir\$scene"
    $sceneData = "$projectDir\$dataDir\$scene"
    $renderDir = "$sceneOutput\renders"
    
    & "$condaPath\Scripts\conda.exe" run -n $envName python "$projectDir\scripts\train_scene.py" `
        --scene_dir $sceneData `
        --model_path $sceneOutput `
        --config "$projectDir\$configFile"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Training failed for $scene, skipping render..."
        continue
    }
    
    Write-Host "RENDERING: $scene"
    New-Item -ItemType Directory -Path $renderDir -Force | Out-Null
    
    & "$condaPath\Scripts\conda.exe" run -n $envName python "$projectDir\scripts\render_test.py" `
        --scene_dir $sceneData `
        --model_path $sceneOutput `
        --output_dir $renderDir
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Render failed for $scene"
        continue
    }
    
    Write-Host "DONE: $scene"
}

Write-Host ""
Write-Host "All scenes processed. Creating submission..."
& "$condaPath\python.exe" python "$projectDir\scripts\make_submission.py" `
    --output_dir "$projectDir\$outputDir" `
    --private_dir "$projectDir\$dataDir" `
    --zip_name "submission_production.zip"

Write-Host "Pipeline complete!"
