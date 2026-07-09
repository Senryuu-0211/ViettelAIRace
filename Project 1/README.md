# VAR 2026 - Digital Twin cho trạm BTS (Vòng 1)

## 1. Tổng Quan Bài Toán

- **Nhiệm vụ**: Novel View Synthesis (NVS) — sinh ảnh RGB tại góc nhìn chưa xuất hiện từ tập ảnh đa góc nhìn.
- **Lĩnh vực**: Computer Vision, 3D Vision, Neural Rendering, Digital Twin.
- **Đối tượng**: Trạm BTS, công trình hạ tầng (dữ liệu từ Drone/Camera cầm tay).
- **Baseline**: 3D Gaussian Splatting (3DGS) — https://github.com/graphdeco-inria/gaussian-splatting

---

## 2. Cấu Trúc Dữ Liệu

```
VAI_NVS_DATA/phase1/
├── public_set/                     # 5 scenes (có ground truth để validate)
│   ├── hcm0031/
│   ├── hcm0034/
│   ├── HCM0181/
│   ├── HCM0193/
│   └── HCM0204/
├── private_set1/                   # 8 scenes (không có ground truth, dùng để submission)
│   ├── HCM0249/
│   ├── HCM0254/
│   ├── HCM0276/
│   ├── HCM1439/
│   ├── HNI0131/
│   ├── HNI0265/
│   ├── HNI0366/
│   └── HNI0437/
```

Mỗi scene có cấu trúc:

```
{scene_name}/
├── train/
│   ├── images/                     # 150-300 ảnh RGB huấn luyện
│   └── sparse/0/                   # COLMAP (cameras.bin, images.bin, points3D.bin)
└── test/
    ├── images/                     # Ground truth (chỉ có ở public_set)
    └── test_poses.csv              # Camera poses cần render
```

### test_poses.csv

| Cột | Ý nghĩa |
|---|---|
| `image_name` | Tên file ảnh đầu ra |
| `qw, qx, qy, qz` | Quaternion rotation (COLMAP) |
| `tx, ty, tz` | Camera translation |
| `fx, fy` | Focal length |
| `cx, cy` | Principal point |
| `width, height` | Kích thước ảnh đầu ra |

---

## 3. Tiêu Chí Đánh Giá

| Metric | Ý nghĩa | Hướng tối ưu |
|---|---|---|
| **LPIPS** | Độ tương đồng cảm quan (Deep Learning) | Càng thấp càng tốt |
| **SSIM** | Độ tương đồng cấu trúc | Càng cao càng tốt |
| **PSNR_norm** | Sai số pixel chuẩn hóa [0,1] | Càng cao càng tốt |

**Công thức**:

```
Score = 0.4 × (1 - LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
```

---

## 4. Yêu Cầu Submission

File nén **submission_round1.zip**, cấu trúc:

```
submission_round1.zip
├── scene_001/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── scene_002/
│   └── ...
└── ...
```

**Yêu cầu**:
- Kích thước ảnh: đúng `width × height` trong `test_poses.csv`
- Tên file: đúng `image_name` trong `test_poses.csv`
- Đầy đủ toàn bộ pose của toàn bộ scene

---

## 5. Timeline

| Mốc | Sự kiện |
|---|---|
| 02/07/2026 | Công bố private test #1 |
| **30/07/2026** | **Deadline submission** |

---

## 6. Quy Định

- **KHÔNG dùng dữ liệu ngoài** (ảnh/video/3D khác, không chỉnh tay Photoshop)
- **Tự động hóa 100%** — ảnh sinh ra hoàn toàn bằng code/model AI
- **Reproducibility** — nếu đạt giải cao, phải cung cấp source code, config, checkpoint, training logs

---

## 7. Pipeline Dự Kiến

```
1. for each scene:
     ├── convert_poses.py       # CSV → camera matrices 4×4
     ├── train_scene.py         # Huấn luyện 3DGS
     ├── render_test.py         # Render test views → PNG
     └── evaluate.py            # Tính metrics (chỉ public set)
2. make_submission.py           # Đóng gói submission_round1.zip
```

### Cấu trúc code dự kiến:

```
Project 1/
├── VAI_NVS_DATA/               # Dữ liệu thô (gitignored)
├── gaussian-splatting/         # Clone baseline 3DGS
├── scripts/                    # Script tùy chỉnh
│   ├── convert_poses.py
│   ├── train_scene.py
│   ├── render_test.py
│   ├── evaluate.py
│   ├── make_submission.py
│   └── utils/
│       └── colmap_utils.py
├── configs/
│   └── default.yaml
├── output/
│   ├── checkpoints/
│   ├── logs/
│   └── submissions/
├── requirements.txt
└── README.md
```

### Hardware tham khảo

- GPU: 1× RTX A4000 (20 GB VRAM)
- CPU: 4–8 cores
- RAM: 16–32 GB

### Số lượng scene

| Tập | Số scene | Có ground truth |
|---|---|---|
| public_set | 5 | Có |
| private_set1 | 8 | Không |
| **Tổng** | **13** | |
