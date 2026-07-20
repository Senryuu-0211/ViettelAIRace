# ============================================================
# Build & Push AWQ INT4 Docker Image
# ============================================================
# KHÔNG CẦN GPU! Model AWQ đã có sẵn trên HuggingFace.
# Chỉ cần Docker Desktop + Internet.
#
# Cách dùng:
#   1. Sửa biến $DOCKERHUB_USER bên dưới
#   2. Chạy: .\build_and_push.ps1
# ============================================================

$DOCKERHUB_USER = "YOUR_DOCKERHUB_USERNAME"   # <-- SỬA Ở ĐÂY
$IMAGE_NAME = "vllm-qwen-awq"
$IMAGE_TAG = "latest"
$FULL_IMAGE = "${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

if ($DOCKERHUB_USER -eq "YOUR_DOCKERHUB_USERNAME") {
    Write-Host "ERROR: Hãy sửa biến DOCKERHUB_USER trong script trước!" -ForegroundColor Red
    exit 1
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Build AWQ INT4 Docker Image"               -ForegroundColor Cyan
Write-Host "  Image: $FULL_IMAGE"                         -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# --- Bước 1: Build Docker image (sẽ tự download AWQ model từ HuggingFace) ---
Write-Host "`n[Step 1/2] Building Docker image..." -ForegroundColor Yellow
Write-Host "  (Sẽ download model AWQ ~1.5GB từ HuggingFace trong lúc build)" -ForegroundColor Gray
docker build -t $FULL_IMAGE .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  Build thành công!" -ForegroundColor Green

# --- Bước 2: Push to Docker Hub ---
Write-Host "`n[Step 2/2] Pushing to Docker Hub..." -ForegroundColor Yellow
Write-Host "  (Đảm bảo đã đăng nhập: docker login)" -ForegroundColor Gray
docker push $FULL_IMAGE
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker push failed! Hãy chạy 'docker login' trước." -ForegroundColor Red
    exit 1
}

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "  DONE! Image: $FULL_IMAGE"                    -ForegroundColor Green
Write-Host ""                                              -ForegroundColor Green
Write-Host "  Tiếp theo:"                                  -ForegroundColor Green
Write-Host "  1. Mở docker-compose-awq.yml"                -ForegroundColor Green
Write-Host "  2. Thay <YOUR_DOCKERHUB_USERNAME>"           -ForegroundColor Green
Write-Host "  3. Submit lên Portal BTC"                    -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
