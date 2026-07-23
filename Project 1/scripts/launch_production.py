"""
Production pipeline launcher. Run with:
    conda run -n 3dgs_vai python scripts/launch_production.py
"""
import subprocess, sys, os, time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "VAI_NVS_DATA_ROUND2_UNDIST")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output_round2_prod")
CONFIG = os.path.join(PROJECT_DIR, "configs", "production.yaml")

PYTHON = sys.executable

scenes = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))

print(f"Scenes: {scenes}")
print(f"Python: {PYTHON}")
print(f"Config: {CONFIG}")
print(f"Data:   {DATA_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)

for scene in scenes:
    scene_out = os.path.join(OUTPUT_DIR, scene)
    scene_data = os.path.join(DATA_DIR, scene)
    renders = os.path.join(scene_out, "renders")

    log_file = os.path.join(scene_out, "train_log.txt")
    os.makedirs(scene_out, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] TRAINING: {scene} (50K iters)")
    print(f"{'='*60}")

    train_cmd = [
        PYTHON, os.path.join(PROJECT_DIR, "scripts", "train_scene.py"),
        "--scene_dir", scene_data,
        "--model_path", scene_out,
        "--config", CONFIG,
    ]

    with open(log_file, "w") as log:
        result = subprocess.run(train_cmd, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"ERROR: Training failed for {scene} (exit {result.returncode})")
        print(f"Log: {log_file}")
        continue

    print(f"[{time.strftime('%H:%M:%S')}] RENDERING: {scene}")

    os.makedirs(renders, exist_ok=True)
    render_cmd = [
        PYTHON, os.path.join(PROJECT_DIR, "scripts", "render_test.py"),
        "--scene_dir", scene_data,
        "--model_path", scene_out,
        "--output_dir", renders,
    ]

    with open(os.path.join(scene_out, "render_log.txt"), "w") as log:
        result = subprocess.run(render_cmd, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"ERROR: Render failed for {scene}")
        continue

    print(f"[{time.strftime('%H:%M:%S')}] DONE: {scene}")

# All done, create submission
print(f"\n{'='*60}")
print("All scenes done. Creating submission zip...")

sub_cmd = [
    PYTHON, os.path.join(PROJECT_DIR, "scripts", "make_submission.py"),
    "--output_dir", OUTPUT_DIR,
    "--private_dir", DATA_DIR,
    "--zip_name", "submission_production.zip",
]
subprocess.run(sub_cmd, cwd=PROJECT_DIR)

print(f"\nPipeline complete at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Submission: {os.path.join(OUTPUT_DIR, 'submission_production.zip')}")
