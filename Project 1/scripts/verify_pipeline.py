"""
Comprehensive verification script for the VAI NVS Pipeline.
Tests all scripts for logical correctness WITHOUT requiring GPU/CUDA.
"""
import os
import sys
import struct
import csv
import math
import importlib.util

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "VAI_NVS_DATA", "phase1")
GS_DIR = os.path.join(PROJECT_DIR, "gaussian-splatting")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

errors = []
warnings = []

def check(condition, msg_pass, msg_fail, is_warning=False):
    if condition:
        print(f"  {PASS} {msg_pass}")
    else:
        if is_warning:
            print(f"  {WARN} {msg_fail}")
            warnings.append(msg_fail)
        else:
            print(f"  {FAIL} {msg_fail}")
            errors.append(msg_fail)

# ============================================================
# 1. Check Project Structure
# ============================================================
print("=" * 60)
print("1. PROJECT STRUCTURE")
print("=" * 60)

check(os.path.exists(os.path.join(PROJECT_DIR, "gaussian-splatting", "train.py")),
      "3DGS repo cloned with train.py",
      "3DGS train.py NOT FOUND - clone may have failed")

check(os.path.exists(os.path.join(PROJECT_DIR, "gaussian-splatting", "submodules", "diff-gaussian-rasterization")),
      "diff-gaussian-rasterization submodule present",
      "diff-gaussian-rasterization submodule MISSING")

check(os.path.exists(os.path.join(PROJECT_DIR, "gaussian-splatting", "submodules", "simple-knn")),
      "simple-knn submodule present",
      "simple-knn submodule MISSING")

check(os.path.exists(os.path.join(PROJECT_DIR, "environment.yml")),
      "environment.yml exists",
      "environment.yml MISSING")

check(os.path.exists(os.path.join(PROJECT_DIR, "configs", "default.yaml")),
      "configs/default.yaml exists",
      "configs/default.yaml MISSING")

for script in ["train_scene.py", "render_test.py", "evaluate.py", "make_submission.py", "run_pipeline.bat", "sanity_check.py"]:
    check(os.path.exists(os.path.join(PROJECT_DIR, "scripts", script)),
          f"scripts/{script} exists",
          f"scripts/{script} MISSING")

# ============================================================
# 2. Check Dataset Structure
# ============================================================
print("\n" + "=" * 60)
print("2. DATASET STRUCTURE")
print("=" * 60)

public_scenes = ["hcm0031", "hcm0034", "HCM0181", "HCM0193", "HCM0204"]
private_scenes = ["HCM0249", "HCM0254", "HCM0276", "HCM1439", "HNI0131", "HNI0265", "HNI0366", "HNI0437"]

scene_stats = []

for subset, scenes in [("public_set", public_scenes), ("private_set1", private_scenes)]:
    for scene in scenes:
        scene_dir = os.path.join(DATA_DIR, subset, scene)
        exists = os.path.exists(scene_dir)
        check(exists, f"{subset}/{scene} exists", f"{subset}/{scene} MISSING")
        
        if not exists:
            continue
        
        # Check train/images
        train_images_dir = os.path.join(scene_dir, "train", "images")
        if os.path.exists(train_images_dir):
            n_train = len([f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            n_train = 0
        
        # Check sparse/0
        sparse_dir = os.path.join(scene_dir, "train", "sparse", "0")
        has_cameras = os.path.exists(os.path.join(sparse_dir, "cameras.bin"))
        has_images_bin = os.path.exists(os.path.join(sparse_dir, "images.bin"))
        has_points3d = os.path.exists(os.path.join(sparse_dir, "points3D.bin"))
        has_points3d_ply = os.path.exists(os.path.join(sparse_dir, "points3D.ply"))
        
        # Check test_poses.csv
        csv_path = os.path.join(scene_dir, "test", "test_poses.csv")
        has_csv = os.path.exists(csv_path)
        n_test_poses = 0
        if has_csv:
            with open(csv_path, 'r') as f:
                n_test_poses = sum(1 for _ in f) - 1  # minus header
        
        # Check test/images (ground truth)
        test_images_dir = os.path.join(scene_dir, "test", "images")
        n_test_gt = 0
        if os.path.exists(test_images_dir):
            n_test_gt = len([f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        scene_stats.append({
            "scene": f"{subset}/{scene}",
            "n_train": n_train,
            "n_test_poses": n_test_poses,
            "n_test_gt": n_test_gt,
            "has_sparse": has_cameras and has_images_bin and has_points3d,
            "has_ply": has_points3d_ply,
        })
        
        check(has_cameras and has_images_bin and has_points3d,
              f"  {scene}: COLMAP sparse OK (cameras.bin + images.bin + points3D.bin)",
              f"  {scene}: COLMAP sparse INCOMPLETE", is_warning=not (has_cameras and has_images_bin and has_points3d))

# ============================================================
# 3. CRITICAL: Camera Model Compatibility Check
# ============================================================
print("\n" + "=" * 60)
print("3. CAMERA MODEL COMPATIBILITY (CRITICAL)")
print("=" * 60)

CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}

# 3DGS only supports SIMPLE_PINHOLE and PINHOLE
SUPPORTED_BY_3DGS = {"SIMPLE_PINHOLE", "PINHOLE"}

for subset, scenes in [("public_set", public_scenes), ("private_set1", private_scenes)]:
    for scene in scenes:
        cameras_path = os.path.join(DATA_DIR, subset, scene, "train", "sparse", "0", "cameras.bin")
        if not os.path.exists(cameras_path):
            continue
        try:
            with open(cameras_path, "rb") as fid:
                num_cameras = struct.unpack("<Q", fid.read(8))[0]
                for _ in range(num_cameras):
                    props = struct.unpack("<iiQQ", fid.read(24))
                    model_id = props[1]
                    model_name, num_params = CAMERA_MODEL_IDS.get(model_id, ("UNKNOWN", 0))
                    width, height = props[2], props[3]
                    # skip params
                    fid.read(8 * num_params)
                    
                    is_supported = model_name in SUPPORTED_BY_3DGS
                    check(is_supported,
                          f"{scene}: Camera model = {model_name} ({width}x{height})",
                          f"{scene}: Camera model = {model_name} -- NOT SUPPORTED by 3DGS! "
                          f"3DGS only supports SIMPLE_PINHOLE and PINHOLE. "
                          f"Must undistort images first or patch dataset_readers.py!")
        except Exception as e:
            check(False, "", f"Failed to read cameras.bin for {scene}: {e}")

# ============================================================
# 4. Check test_poses.csv format consistency
# ============================================================
print("\n" + "=" * 60)
print("4. TEST POSES CSV FORMAT")
print("=" * 60)

expected_cols = ['image_name', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'fx', 'fy', 'cx', 'cy', 'width', 'height']

for subset, scenes in [("public_set", public_scenes), ("private_set1", private_scenes)]:
    for scene in scenes:
        csv_path = os.path.join(DATA_DIR, subset, scene, "test", "test_poses.csv")
        if not os.path.exists(csv_path):
            check(False, "", f"{scene}: test_poses.csv NOT FOUND")
            continue
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            header = [h.strip() for h in header]
            
            cols_match = header == expected_cols
            check(cols_match,
                  f"{scene}: CSV columns match expected format ({len(header)} cols)",
                  f"{scene}: CSV columns MISMATCH. Got: {header}")

# ============================================================
# 5. Check render_test.py logic
# ============================================================
print("\n" + "=" * 60)
print("5. RENDER_TEST.PY LOGIC REVIEW")
print("=" * 60)

# The key issue: render_test.py uses MiniCam which doesn't have all attributes
# that the 3DGS renderer expects. Let's check what the render function needs.
render_py = os.path.join(GS_DIR, "gaussian_renderer", "__init__.py")
if os.path.exists(render_py):
    with open(render_py, 'r') as f:
        render_src = f.read()
    
    # Check what attributes render() accesses on the viewpoint_camera
    cam_attrs_used = []
    for attr in ["image_height", "image_width", "FoVx", "FoVy", "world_view_transform", 
                 "full_proj_transform", "camera_center", "znear", "zfar",
                 "original_image", "uid", "exposure_mapping"]:
        if f"viewpoint_camera.{attr}" in render_src or f"viewpoint_camera.{attr}" in render_src:
            cam_attrs_used.append(attr)
    
    # MiniCam in our render_test.py has:
    minicam_attrs = ["image_width", "image_height", "FoVy", "FoVx", "znear", "zfar", 
                     "world_view_transform", "full_proj_transform", "camera_center"]
    
    missing = set(cam_attrs_used) - set(minicam_attrs)
    check(len(missing) == 0,
          f"MiniCam has all required attributes for render(): {cam_attrs_used}",
          f"MiniCam MISSING attributes needed by render(): {missing}")

# ============================================================
# 6. Check train_scene.py generates correct command
# ============================================================
print("\n" + "=" * 60)
print("6. TRAIN_SCENE.PY COMMAND VERIFICATION")
print("=" * 60)

# Verify that train.py accepts '--data_device' flag (important for VRAM saving)
with open(os.path.join(GS_DIR, "arguments", "__init__.py"), 'r') as f:
    args_src = f.read()

check("data_device" in args_src,
      "3DGS supports --data_device flag (for CPU data loading)",
      "3DGS does NOT have --data_device flag")

# Check if our train_scene.py passes --data_device
with open(os.path.join(PROJECT_DIR, "scripts", "train_scene.py"), 'r') as f:
    train_src = f.read()

check("data_device" in train_src,
      "train_scene.py passes --data_device",
      "train_scene.py does NOT pass --data_device (may cause OOM on limited VRAM)", is_warning=True)

check("--disable_viewer" in train_src or "disable_viewer" in train_src,
      "train_scene.py passes --disable_viewer",
      "train_scene.py does NOT pass --disable_viewer (viewer may fail without GUI deps)", is_warning=True)

# ============================================================
# 7. Check output image naming
# ============================================================
print("\n" + "=" * 60)
print("7. OUTPUT IMAGE NAMING")
print("=" * 60)

# The test_poses.csv has image_name like 'DJI_20241227155343_0023_V.JPG'
# render_test.py saves as the image_name directly -> this will produce .JPG files
# But torchvision.utils.save_image always saves as the given extension
# So if image_name is .JPG, it will save as .JPG (which is actually PNG format internally from torchvision)
# This could be problematic - submission might want .png files

csv_path = os.path.join(DATA_DIR, "public_set", "hcm0031", "test", "test_poses.csv")
if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader)
        img_name = first_row[0]
        ext = os.path.splitext(img_name)[1]
        check(ext.lower() == ".png",
              f"Test image names use .png extension",
              f"Test image names use '{ext}' extension. "
              f"torchvision.save_image uses the given extension - verify if submission needs .png", is_warning=True)

# ============================================================
# 8. Summary Statistics
# ============================================================
print("\n" + "=" * 60)
print("8. DATASET SUMMARY")
print("=" * 60)

print(f"  {'Scene':<25} {'Train':>6} {'Test Poses':>11} {'Test GT':>8} {'COLMAP':>7} {'PLY':>4}")
print("  " + "-" * 65)
for s in scene_stats:
    print(f"  {s['scene']:<25} {s['n_train']:>6} {s['n_test_poses']:>11} {s['n_test_gt']:>8} {'OK' if s['has_sparse'] else 'MISSING':>7} {'Yes' if s['has_ply'] else 'No':>4}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

if errors:
    print(f"\n{FAIL} {len(errors)} ERROR(s) found:")
    for i, e in enumerate(errors, 1):
        print(f"  {i}. {e}")
else:
    print(f"\n{PASS} No critical errors found.")

if warnings:
    print(f"\n{WARN} {len(warnings)} WARNING(s):")
    for i, w in enumerate(warnings, 1):
        print(f"  {i}. {w}")

print()
