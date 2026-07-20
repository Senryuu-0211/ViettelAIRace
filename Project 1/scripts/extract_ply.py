"""Extract point_cloud.ply from a .pth checkpoint without calling restore/training_setup."""
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gaussian-splatting")))
from scene import GaussianModel

if len(sys.argv) < 4:
    print("Usage: python extract_ply.py <checkpoint.pth> <output_ply_path> <sh_degree>")
    sys.exit(1)

checkpoint_path = sys.argv[1]
output_ply = sys.argv[2]
sh_degree = int(sys.argv[3])

print(f"Loading checkpoint: {checkpoint_path}")
model_params, iter_num = torch.load(checkpoint_path, map_location="cpu")

# Unpack model_params directly (matches capture() order)
(active_sh_degree,
 _xyz,
 _features_dc,
 _features_rest,
 _scaling,
 _rotation,
 _opacity,
 max_radii2D,
 xyz_gradient_accum,
 denom,
 opt_dict,
 spatial_lr_scale) = model_params

# Create GaussianModel and manually assign tensors (skip training_setup entirely)
gaussians = GaussianModel(sh_degree)
gaussians.active_sh_degree = active_sh_degree
gaussians._xyz = _xyz
gaussians._features_dc = _features_dc
gaussians._features_rest = _features_rest
gaussians._scaling = _scaling
gaussians._rotation = _rotation
gaussians._opacity = _opacity
gaussians.max_radii2D = max_radii2D
gaussians.spatial_lr_scale = spatial_lr_scale

os.makedirs(os.path.dirname(output_ply), exist_ok=True)
gaussians.save_ply(output_ply)
print(f"OK: Extracted iteration {iter_num} -> {output_ply}")
print(f"    Points: {_xyz.shape[0]}")
