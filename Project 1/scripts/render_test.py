import os
import sys
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision

# Add gaussian-splatting to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gaussian-splatting")))

from scene import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from argparse import ArgumentParser

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def main():
    parser = ArgumentParser(description="Render test views for a scene")
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--iteration", type=int, default=None, help="Specific iteration to load (default: latest)")
    
    args = parser.parse_args()
    
    csv_path = os.path.join(args.scene_dir, "test", "test_poses.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    gaussians = GaussianModel(args.sh_degree)
    
    # Find the latest iteration checkpoint
    checkpoints_dir = os.path.join(args.model_path, "point_cloud")
    if not os.path.exists(checkpoints_dir):
        print(f"Error: No point_cloud dir in {args.model_path}")
        sys.exit(1)
        
    iters = [int(d.split("_")[-1]) for d in os.listdir(checkpoints_dir) if d.startswith("iteration_")]
    if len(iters) == 0:
        print("No iterations found")
        sys.exit(1)
    
    if args.iteration is not None:
        latest_iter = args.iteration
    else:
        latest_iter = max(iters)
    checkpoint_path = os.path.join(checkpoints_dir, f"iteration_{latest_iter}", "point_cloud.ply")
    
    print(f"Loading checkpoint from iteration {latest_iter}")
    gaussians.load_ply(checkpoint_path)
    
    # Create background
    bg_color = [1,1,1] if args.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Pipeline params
    class PipelineParams:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
            self.antialiasing = False
    pipeline = PipelineParams()
    
    print("Rendering test images...")
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_name = row['image_name']
            qw, qx, qy, qz = row['qw'], row['qx'], row['qy'], row['qz']
            tx, ty, tz = row['tx'], row['ty'], row['tz']
            fx, fy = row['fx'], row['fy']
            width, height = int(row['width']), int(row['height'])
            
            # Convert
            R = qvec2rotmat([qw, qx, qy, qz])
            T = np.array([tx, ty, tz])
            
            FoVx = focal2fov(fx, width)
            FoVy = focal2fov(fy, height)
            
            # create view and proj matrices
            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0,0.0,0.0]), 1.0)).float().transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            
            cam = MiniCam(width, height, FoVy, FoVx, 0.01, 100.0, world_view_transform, full_proj_transform)
            
            render_pkg = render(cam, gaussians, pipeline, background)
            rendered_image = render_pkg["render"]
            
            # Save
            out_path = os.path.join(args.output_dir, image_name)
            torchvision.utils.save_image(rendered_image, out_path)
            
    print("Done!")

if __name__ == "__main__":
    main()
