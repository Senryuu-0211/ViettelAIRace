import os
import sys
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision
from argparse import ArgumentParser, Namespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Scaffold-GS")))

from scene import GaussianModel
from scene.cameras import MiniCam
from gaussian_renderer import render, prefilter_voxel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

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
    return 2 * math.atan(pixels / (2 * focal))

def main():
    parser = ArgumentParser(description="Render test views using Scaffold-GS")
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--feat_dim", type=int, default=32)
    parser.add_argument("--n_offsets", type=int, default=10)
    parser.add_argument("--voxel_size", type=float, default=0.001)
    parser.add_argument("--update_depth", type=int, default=3)
    parser.add_argument("--update_init_factor", type=int, default=16)
    parser.add_argument("--update_hierachy_factor", type=int, default=4)

    args = parser.parse_args()

    csv_path = os.path.join(args.scene_dir, "test", "test_poses.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # Read training config
    cfg_path = os.path.join(args.model_path, "cfg_args")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = eval(f.read(), {"Namespace": Namespace})
    else:
        cfg = None

    gaussians = GaussianModel(
        feat_dim=getattr(cfg, 'feat_dim', args.feat_dim) if cfg else args.feat_dim,
        n_offsets=getattr(cfg, 'n_offsets', args.n_offsets) if cfg else args.n_offsets,
        voxel_size=getattr(cfg, 'voxel_size', args.voxel_size) if cfg else args.voxel_size,
        update_depth=getattr(cfg, 'update_depth', args.update_depth) if cfg else args.update_depth,
        update_init_factor=getattr(cfg, 'update_init_factor', args.update_init_factor) if cfg else args.update_init_factor,
        update_hierachy_factor=getattr(cfg, 'update_hierachy_factor', args.update_hierachy_factor) if cfg else args.update_hierachy_factor,
        use_feat_bank=getattr(cfg, 'use_feat_bank', False) if cfg else False,
        appearance_dim=getattr(cfg, 'appearance_dim', 32) if cfg else 32,
        ratio=getattr(cfg, 'ratio', 1) if cfg else 1,
        add_opacity_dist=getattr(cfg, 'add_opacity_dist', False) if cfg else False,
        add_cov_dist=getattr(cfg, 'add_cov_dist', False) if cfg else False,
        add_color_dist=getattr(cfg, 'add_color_dist', False) if cfg else False,
    )

    checkpoints_dir = os.path.join(args.model_path, "point_cloud")
    if args.iteration == -1:
        iters = [int(d.split("_")[-1]) for d in os.listdir(checkpoints_dir) if d.startswith("iteration_")]
        iteration = max(iters)
    else:
        iteration = args.iteration

    ply_path = os.path.join(checkpoints_dir, f"iteration_{iteration}", "point_cloud.ply")
    mlp_dir = os.path.join(checkpoints_dir, f"iteration_{iteration}")

    print(f"Loading model from iteration {iteration}")
    gaussians.load_ply_sparse_gaussian(ply_path)
    gaussians.load_mlp_checkpoints(mlp_dir)
    gaussians.eval()

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    class Pipe:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
    pipeline = Pipe()

    print(f"Rendering {len(df)} test views...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_name = row["image_name"]
            qw, qx, qy, qz = row["qw"], row["qx"], row["qy"], row["qz"]
            tx, ty, tz = row["tx"], row["ty"], row["tz"]
            fx, fy = row["fx"], row["fy"]
            width, height = int(row["width"]), int(row["height"])

            R = qvec2rotmat([qw, qx, qy, qz])
            T = np.array([tx, ty, tz])

            FoVx = focal2fov(fx, width)
            FoVy = focal2fov(fy, height)

            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).float().transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

            cam = MiniCam(width, height, FoVy, FoVx, 0.01, 100.0, world_view_transform, full_proj_transform)
            cam.uid = 0

            voxel_visible_mask = prefilter_voxel(cam, gaussians, pipeline, background)
            render_pkg = render(cam, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)

            out_path = os.path.join(args.output_dir, image_name)
            torchvision.utils.save_image(rendered_image, out_path)

    print(f"Done! Rendered {len(df)} images to {args.output_dir}")

if __name__ == "__main__":
    main()
