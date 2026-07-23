"""
Resume training from existing 7K checkpoints.
Train additional iterations on ORIGINAL data (distortion is negligible, k=0.009).
Total target: 25K iterations (= 18K more from 7K checkpoint).
"""
import subprocess, sys, os, time, shutil, glob

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "VAI_NVS_DATA_ROUND2")  # ORIGINAL data
OLD_OUTPUT = os.path.join(PROJECT_DIR, "output_round2")        # existing 7K checkpoints  
NEW_OUTPUT = os.path.join(PROJECT_DIR, "output_prod")
CONFIG = os.path.join(PROJECT_DIR, "configs", "production.yaml")

PYTHON = sys.executable
TOTAL_ITERS = 25000
RESUME_FROM = 7000
ADDITIONAL = TOTAL_ITERS - RESUME_FROM  # 18000

GS_TRAIN = os.path.join(PROJECT_DIR, "gaussian-splatting", "train.py")

scenes = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))

for scene in scenes:
    old_chkpnt = os.path.join(OLD_OUTPUT, scene, f"chkpnt{RESUME_FROM}.pth")
    if not os.path.exists(old_chkpnt):
        print(f"[{scene}] No checkpoint at {old_chkpnt}, skipping")
        continue

    new_out = os.path.join(NEW_OUTPUT, scene)
    os.makedirs(new_out, exist_ok=True)

    # Copy checkpoint to new output dir
    dst_chkpnt = os.path.join(new_out, f"chkpnt{RESUME_FROM}.pth")
    if not os.path.exists(dst_chkpnt):
        shutil.copy2(old_chkpnt, dst_chkpnt)
        print(f"[{scene}] Copied checkpoint from iteration {RESUME_FROM}")

    source_path = os.path.join(DATA_DIR, scene, "train")
    log_file = os.path.join(new_out, "train_log.txt")

    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] RESUME {scene}: 7K → 25K ({ADDITIONAL} more iters)")
    print(f"Source: {source_path}")
    print(f"Output: {new_out}")
    print(f"{'='*60}")

    cmd = [
        PYTHON, GS_TRAIN,
        "-s", source_path,
        "-m", new_out,
        "--start_checkpoint", dst_chkpnt,
        "--iterations", str(TOTAL_ITERS),
        "--disable_viewer",
        "--data_device", "cpu",
        "--eval",
        "--densify_until_iter", str(TOTAL_ITERS // 2),
        "--densify_grad_threshold", "0.0001",
        "--opacity_reset_interval", "5000",
        "--densify_from_iter", "500",
    ]

    print(f"Cmd: {' '.join(cmd)}")

    with open(log_file, "w") as log:
        result = subprocess.run(cmd, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"  ERROR: exit={result.returncode}")
        continue

    # Render
    renders = os.path.join(new_out, "renders")
    os.makedirs(renders, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] RENDERING {scene}")

    render_cmd = [
        PYTHON, os.path.join(PROJECT_DIR, "scripts", "render_test.py"),
        "--scene_dir", os.path.join(DATA_DIR, scene),
        "--model_path", new_out,
        "--output_dir", renders,
    ]
    with open(os.path.join(new_out, "render_log.txt"), "w") as log:
        result = subprocess.run(render_cmd, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"  ERROR rendering {scene}")
        continue

    print(f"[{time.strftime('%H:%M:%S')}] DONE: {scene}")

# Submission
print(f"\n{'='*60}")
print("Creating submission...")
subprocess.run([
    PYTHON, os.path.join(PROJECT_DIR, "scripts", "make_submission.py"),
    "--output_dir", NEW_OUTPUT,
    "--private_dir", DATA_DIR,
    "--zip_name", "submission_prod.zip",
], cwd=PROJECT_DIR)
print("Done!")
