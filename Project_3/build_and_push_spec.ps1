# ============================================================
# Build & Push image SPEC DECODING (vLLM v0.25.1 + AWQ INT4 LFM2.5)
# KHÔNG CẦN GPU để build. Cần Docker + `docker login`.
#   1. Sửa $DOCKERHUB_USER
#   2. .\build_and_push_spec.ps1
#   3. Sửa image trong docker-compose-spec-v2.yml cho khớp
# ⚠️ Sau khi build: PHẢI test trên H200 + verify output KHÔNG corrupt
#    (so với greedy) TRƯỚC khi nộp — xem SPEC_DECODING_FINDINGS.md mục 4.
# ============================================================
$DOCKERHUB_USER = "YOUR_DOCKERHUB_USERNAME"   # <-- SỬA Ở ĐÂY
$FULL_IMAGE = "${DOCKERHUB_USER}/vllm-lfm2-awq-spec:v0251"   # PIN tag (không :latest)

if ($DOCKERHUB_USER -eq "YOUR_DOCKERHUB_USERNAME") {
    Write-Host "ERROR: Sửa DOCKERHUB_USER trước!" -ForegroundColor Red; exit 1
}

Write-Host "[1/2] Building $FULL_IMAGE ..." -ForegroundColor Yellow
docker build -f Dockerfile.spec -t $FULL_IMAGE .
if ($LASTEXITCODE -ne 0) { Write-Host "Build FAILED" -ForegroundColor Red; exit 1 }

Write-Host "[2/2] Pushing (cần docker login) ..." -ForegroundColor Yellow
docker push $FULL_IMAGE
if ($LASTEXITCODE -ne 0) { Write-Host "Push FAILED — chạy 'docker login' trước" -ForegroundColor Red; exit 1 }

Write-Host "DONE: $FULL_IMAGE" -ForegroundColor Green
