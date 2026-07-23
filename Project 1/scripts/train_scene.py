import os
import sys
import subprocess
import yaml
import argparse
import glob
import re

def main():
    parser = argparse.ArgumentParser(description="Train 3DGS on a specific scene")
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to the scene directory (e.g. public_set/hcm0031)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save the output model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    scene_dir = args.scene_dir
    model_path = args.model_path
    config_path = args.config
    
    # Check if scene exists
    if not os.path.exists(scene_dir):
        print(f"Error: Scene directory {scene_dir} does not exist.")
        sys.exit(1)
        
    # Read config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # The source for 3DGS is the train/ folder inside the scene
    source_path = os.path.join(scene_dir, "train")
    
    print(f"--- Training 3DGS on Scene: {os.path.basename(scene_dir)} ---")
    print(f"Source: {source_path}")
    print(f"Output: {model_path}")
    print(f"Config: {config_path}")
    
    # Build command for gaussian-splatting/train.py
    # Note: 3DGS train.py takes -s for source, -m for model path
    cmd = [
        sys.executable, "gaussian-splatting/train.py",
        "-s", source_path,
        "-m", model_path,
        "--disable_viewer",
        "--data_device", "cpu",
        "--eval",
    ]
    
    # Add hyperparams from config
    if "iterations" in config:
        cmd.extend(["--iterations", str(config["iterations"])])
    if "position_lr_init" in config:
        cmd.extend(["--position_lr_init", str(config["position_lr_init"])])
    if "position_lr_final" in config:
        cmd.extend(["--position_lr_final", str(config["position_lr_final"])])
    if "position_lr_delay_mult" in config:
        cmd.extend(["--position_lr_delay_mult", str(config["position_lr_delay_mult"])])
    if "position_lr_max_steps" in config:
        cmd.extend(["--position_lr_max_steps", str(config["position_lr_max_steps"])])
    if "feature_lr" in config:
        cmd.extend(["--feature_lr", str(config["feature_lr"])])
    if "opacity_lr" in config:
        cmd.extend(["--opacity_lr", str(config["opacity_lr"])])
    if "scaling_lr" in config:
        cmd.extend(["--scaling_lr", str(config["scaling_lr"])])
    if "rotation_lr" in config:
        cmd.extend(["--rotation_lr", str(config["rotation_lr"])])
    if "percent_dense" in config:
        cmd.extend(["--percent_dense", str(config["percent_dense"])])
    if "lambda_dssim" in config:
        cmd.extend(["--lambda_dssim", str(config["lambda_dssim"])])
    if "densification_interval" in config:
        cmd.extend(["--densification_interval", str(config["densification_interval"])])
    if "opacity_reset_interval" in config:
        cmd.extend(["--opacity_reset_interval", str(config["opacity_reset_interval"])])
        
    # Checkpoints interval (every 1000 iterations up to max iterations)
    max_iters = config.get("iterations", 30000)
    checkpoint_iters = [str(i) for i in range(1000, max_iters + 1, 1000)]
    cmd.extend(["--checkpoint_iterations"] + checkpoint_iters)
    
    # Auto-resume logic
    checkpoints = glob.glob(os.path.join(model_path, "chkpnt*.pth"))
    if checkpoints:
        # Extract iteration numbers to find the latest
        latest_chkpnt = None
        max_iter = -1
        for cp in checkpoints:
            match = re.search(r"chkpnt(\d+)\.pth", os.path.basename(cp))
            if match:
                iter_num = int(match.group(1))
                if iter_num > max_iter:
                    max_iter = iter_num
                    latest_chkpnt = cp
        
        if latest_chkpnt:
            print(f"Resuming from checkpoint: {latest_chkpnt}")
            cmd.extend(["--start_checkpoint", latest_chkpnt])

    if "densify_from_iter" in config:
        cmd.extend(["--densify_from_iter", str(config["densify_from_iter"])])
    if "densify_until_iter" in config:
        cmd.extend(["--densify_until_iter", str(config["densify_until_iter"])])
    if "densify_grad_threshold" in config:
        cmd.extend(["--densify_grad_threshold", str(config["densify_grad_threshold"])])
    if "sh_degree" in config:
        cmd.extend(["--sh_degree", str(config["sh_degree"])])
    if "white_background" in config and config["white_background"]:
        cmd.extend(["--white_background"])
        
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
