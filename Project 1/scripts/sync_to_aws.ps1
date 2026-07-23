# Upload Scaffold-GS code + data to AWS g5.xlarge for training
# Usage: .\sync_to_aws.ps1
# Expects viettel.pem in the project root (D:\ViettelAIRace\Project 1\viettel.pem)
$ErrorActionPreference = "Stop"

$PROJECT  = Split-Path -Parent $PSScriptRoot          # D:\...\Project 1
$KEY      = Join-Path $PROJECT "viettel.pem"
$SERVER   = "ec2-user@44.213.131.129"
$STAGE    = Join-Path $env:TEMP "aws_sync_stage"

$folders = @("Scaffold-GS", "scripts", "configs", "VAI_NVS_DATA_ROUND2", "output_scaffold")

Write-Host "=== [1/3] Staging (skip build junk, .git, __pycache__) ==="
if (Test-Path $STAGE) { Remove-Item $STAGE -Recurse -Force }
foreach ($f in $folders) {
    $src = Join-Path $PROJECT $f
    $dst = Join-Path $STAGE $f
    Write-Host "  staging $f ..."
    robocopy $src $dst /E /XD __pycache__ build .git /XF *.pyc *.pyd *.obj *.exp *.lib 2>&1 | Out-Null
    if ($LASTEXITCODE -gt 7) { Write-Host "  ERROR $LASTEXITCODE copying $f"; exit 1 }
}

Write-Host "`n=== [2/3] Uploading (~3.2 GB) ==="
# Create ~/project on remote
ssh -i $KEY -o ConnectTimeout=15 $SERVER "mkdir -p project"
foreach ($f in $folders) {
    Write-Host "  scp $f ..."
    scp -i $KEY -o ConnectTimeout=60 -r (Join-Path $STAGE $f) "${SERVER}:project/" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "  ERROR scp $f"; exit 1 }
}

Write-Host "`n=== [3/3] Fix line endings (setup script must be LF) ==="
ssh -i $KEY $SERVER "sed -i 's/\r$//' project/scripts/setup_aws.sh 2>/dev/null; echo OK"

Write-Host "`n=== DONE. Next: ==="
Write-Host "  ssh -i $KEY $SERVER"
Write-Host "  bash ~/project/scripts/setup_aws.sh"
