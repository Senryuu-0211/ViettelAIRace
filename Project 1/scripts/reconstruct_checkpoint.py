"""
Reconstruct chkpnt7000.pth from point_cloud/iteration_7000/.
Bypasses Scene constructor - just loads model weights + saves captured state.

Usage: conda run -n scaffold_gs python scripts/reconstruct_checkpoint.py HCM0421
"""
import sys, os, torch
from argparse import Namespace

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAFFOLD_DIR = os.path.join(PROJECT_DIR, "Scaffold-GS")
sys.path.insert(0, SCAFFOLD_DIR)

from argparse import ArgumentParser
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, OptimizationParams, PipelineParams

OLD_OUTPUT = os.path.join(PROJECT_DIR, "output_scaffold")
LOAD_ITER = 7000


def reconstruct(scene_name):
    old_dir = os.path.join(OLD_OUTPUT, scene_name)
    pc_dir = os.path.join(old_dir, "point_cloud", f"iteration_{LOAD_ITER}")
    chkpnt_path = os.path.join(old_dir, f"chkpnt{LOAD_ITER}.pth")

    if os.path.exists(chkpnt_path):
        print(f"[{scene_name}] Already exists: {chkpnt_path}")
        return True
    if not os.path.exists(pc_dir):
        print(f"[{scene_name}] No point_cloud at {pc_dir}")
        return False

    # Read original cfg to get model dimensions
    cfg_file = os.path.join(old_dir, "cfg_args")
    if not os.path.exists(cfg_file):
        print(f"[{scene_name}] No cfg_args")
        return False

    with open(cfg_file) as f:
        cfg_str = f.read()
    ns = eval(cfg_str)

    # Manually create model with same architecture
    gaussians = GaussianModel(
        feat_dim=ns.feat_dim,
        n_offsets=ns.n_offsets,
        voxel_size=ns.voxel_size,
        update_depth=ns.update_depth,
        update_init_factor=ns.update_init_factor,
        update_hierachy_factor=ns.update_hierachy_factor,
        use_feat_bank=ns.use_feat_bank,
        appearance_dim=ns.appearance_dim,
        ratio=ns.ratio,
        add_opacity_dist=ns.add_opacity_dist,
        add_cov_dist=ns.add_cov_dist,
        add_color_dist=ns.add_color_dist,
    )

    # Load sparse gaussian (PLY)
    ply_path = os.path.join(pc_dir, "point_cloud.ply")
    if os.path.exists(ply_path):
        gaussians.load_ply_sparse_gaussian(ply_path)
        print(f"[{scene_name}] Loaded PLY: {gaussians.get_anchor.shape[0]} anchors")
    else:
        print(f"[{scene_name}] No point_cloud.ply at {ply_path}")
        return False

    # Load MLP checkpoints
    gaussians.load_mlp_checkpoints(pc_dir)
    print(f"[{scene_name}] Loaded MLP checkpoints")

    # Initialize optimizer (all defaults, then capture)
    parser = ArgumentParser()
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    opt = op.extract(parser.parse_args([]))
    gaussians.training_setup(opt)

    # _local is never initialized when use_feat_bank=False but capture() requires it
    if not hasattr(gaussians, '_local'):
        gaussians._local = torch.empty(0)

    model_params = gaussians.capture()
    torch.save((model_params, LOAD_ITER), chkpnt_path)
    size_mb = os.path.getsize(chkpnt_path) / 1024 / 1024
    print(f"[{scene_name}] Saved: {chkpnt_path} ({size_mb:.1f} MB)")
    return True


if __name__ == "__main__":
    _orig_argv = sys.argv[:]
    if len(_orig_argv) > 1:
        target = _orig_argv[1]
        if target == "--all":
            scenes = sorted(d for d in os.listdir(OLD_OUTPUT)
                          if os.path.isdir(os.path.join(OLD_OUTPUT, d, "point_cloud")))
            for s in scenes:
                reconstruct(s)
        else:
            reconstruct(target)
    else:
        print("Usage: python reconstruct_checkpoint.py HCM0421  or  --all")
