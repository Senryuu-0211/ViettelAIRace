"""
Scaffold-GS resume: load pretrained PLY + MLP, continue training with new config.
Auto-resume: saves PLY+MLP every SAVE_INTERVAL iters via scene.save(); on rerun,
picks up from the highest saved iteration in the output dir.

Usage:
  conda run -n scaffold_gs python scripts/resume_scaffold.py --test_1_scene HCM0421
  conda run -n scaffold_gs python scripts/resume_scaffold.py --all
"""
import sys, os, time, csv, shutil, yaml

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAFFOLD_DIR = os.path.join(PROJECT_DIR, "Scaffold-GS")
sys.path.insert(0, SCAFFOLD_DIR)

import torch, numpy as np
from argparse import ArgumentParser
from random import randint
from tqdm import tqdm

from scene import Scene, GaussianModel
from scene.gaussian_model import inverse_sigmoid
from gaussian_renderer import render, prefilter_voxel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss, ssim

DATA_DIR = os.path.join(PROJECT_DIR, "VAI_NVS_DATA_ROUND2")
OLD_OUTPUT = os.path.join(PROJECT_DIR, "output_scaffold")
NEW_OUTPUT = os.path.join(PROJECT_DIR, "output_scaffold_resume")
LOAD_ITER = 7000
TOTAL_ITERS = 12000
SAVE_INTERVAL = 2000

def get_default_opt(parser, overrides):
    """Parse empty args (all defaults) then override with config values."""
    args = parser.parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def verify_buffers(gaussians, label):
    print(f"\n[BUFFER] {label} n_anchors={gaussians.get_anchor.shape[0]}")
    for name in ["opacity_accum", "offset_gradient_accum", "offset_denom", "anchor_demon"]:
        buf = getattr(gaussians, name, None)
        if buf is None or buf.numel() == 0:
            print(f"  {name}: EMPTY")
        else:
            nz = (buf > 0).sum().item()
            st = "DIRTY" if nz > 0 else "CLEAN"
            print(f"  {name}: mean={buf.mean().item():.6f} max={buf.max().item():.6f} nonzero={nz} {st}")


class AnchorLogger:
    def __init__(self, path):
        self.path = path; self.rows = []; self.prev_n = 0
    def log(self, it, g, phase):
        n = g.get_anchor.shape[0]; d = n - self.prev_n; self.prev_n = n
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        self.rows.append([it, phase, n, d, f"{alloc:.2f}", f"{reserved:.2f}"])
        print(f"  [ANCHOR] {it} {phase}: n={n} d={d:+d} alloc={alloc:.2f}G reserved={reserved:.2f}G")
    def save(self):
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iteration","phase","n_anchors","delta","vram_alloc_gb","vram_reserved_gb"])
            w.writerows(self.rows)


class LossMonitor:
    def __init__(self, iters=300, spike=50):
        self.monitor = iters; self.limit = 1 + spike/100; self.bl1 = None; self.bls = None; self.warned = False
    def check(self, it, l1, ssim_l):
        if it > self.monitor: return
        if self.bl1 is None: self.bl1 = l1; self.bls = ssim_l; print(f"[LOSS] baseline L1={l1:.6f} SSIM_loss={ssim_l:.6f}"); return
        if not self.warned and (l1 > self.bl1*self.limit or ssim_l > self.bls*self.limit):
            self.warned = True
            print(f"\n[SPIKE] iter={it} L1={l1:.6f} (base={self.bl1:.6f}) SSIM_loss={ssim_l:.6f} (base={self.bls:.6f})")


def verify_thresholds(opt):
    dmin = opt.update_interval * opt.success_threshold * 0.5
    print(f"\n[THRESHOLDS] success={opt.success_threshold} grad={opt.densify_grad_threshold} "
          f"update_from={opt.update_from} update_until={opt.update_until} offset_min_denom={dmin:.0f}")


def find_latest_iter(model_path, min_iter):
    """Find highest saved iteration_N (>= min_iter) in model_path/point_cloud."""
    pc_dir = os.path.join(model_path, "point_cloud")
    best = None
    if os.path.isdir(pc_dir):
        for d in os.listdir(pc_dir):
            if d.startswith("iteration_"):
                try:
                    it = int(d.rsplit("_", 1)[1])
                except ValueError:
                    continue
                if it >= min_iter and (best is None or it > best):
                    best = it
    return best


def training_finetune(dataset, opt, pipe, anchor_log, loss_mon, start_iter):
    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size,
        dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor,
        dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio,
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
    )

    # Scene loads PLY from model_path/point_cloud/iteration_{start_iter}/
    scene = Scene(dataset, gaussians, load_iteration=start_iter, shuffle=False)

    if start_iter >= opt.iterations:
        print(f"[SKIP] already at iter {start_iter} >= target {opt.iterations}")
        return

    gaussians.training_setup(opt)
    gaussians.train()  # set MLPs to training mode (render checks this for is_training)

    verify_buffers(gaussians, "after training_setup + PLY load")
    verify_thresholds(opt)
    anchor_log.prev_n = gaussians.get_anchor.shape[0]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    vp_stack = None
    ema = 0.0

    pbar = tqdm(range(start_iter, opt.iterations), desc=f"Finetune {start_iter}->{opt.iterations}")

    for iteration in range(start_iter + 1, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if not vp_stack:
            vp_stack = scene.getTrainCameras().copy()
        vp_cam = vp_stack.pop(randint(0, len(vp_stack) - 1))

        voxel_mask = prefilter_voxel(vp_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        rpkg = render(vp_cam, gaussians, pipe, background, visible_mask=voxel_mask, retain_grad=retain_grad)

        image = rpkg["render"]
        vs_pts = rpkg["viewspace_points"]
        vis = rpkg["visibility_filter"]
        sel = rpkg["selection_mask"]
        scaling = rpkg["scaling"]
        opacity = rpkg["neural_opacity"]

        gt = vp_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt)
        ssim_l = 1.0 - ssim(image, gt)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_l + 0.01 * scaling.prod(dim=1).mean()
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema = 0.4 * loss.item() + 0.6 * ema
            loss_mon.check(iteration, Ll1.item(), ssim_l)

            if iteration % 10 == 0:
                pbar.set_postfix({"Loss": f"{ema:.7f}"})
                pbar.update(10)
            if iteration == opt.iterations:
                pbar.close()

            if iteration < opt.update_until and iteration > opt.start_stat:
                gaussians.training_statis(vs_pts, opacity, vis, sel, voxel_mask)
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(
                        check_interval=opt.update_interval,
                        success_threshold=opt.success_threshold,
                        grad_threshold=opt.densify_grad_threshold,
                        min_opacity=opt.min_opacity)
                    anchor_log.log(iteration, gaussians, "grow+prune")
                    torch.cuda.empty_cache()
            elif iteration == opt.update_until:
                for attr in ["opacity_accum", "offset_gradient_accum", "offset_denom"]:
                    if hasattr(gaussians, attr):
                        delattr(gaussians, attr)
                torch.cuda.empty_cache()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration % SAVE_INTERVAL == 0:
                print(f"\n[CKPT] saving iter {iteration}, Loss={ema:.6f}")
                scene.save(iteration)

    scene.save(opt.iterations)
    anchor_log.save()


def run_scene(scene_name):
    print(f"\n{'='*60}\nTEST: {scene_name}\n{'='*60}")

    scene_data = os.path.join(DATA_DIR, scene_name)
    old_pc = os.path.join(OLD_OUTPUT, scene_name, "point_cloud", f"iteration_{LOAD_ITER}")
    new_out = os.path.join(NEW_OUTPUT, scene_name)

    if not os.path.exists(old_pc):
        print(f"[ERROR] No point_cloud at {old_pc}"); return False

    os.makedirs(new_out, exist_ok=True)
    new_pc = os.path.join(new_out, "point_cloud", f"iteration_{LOAD_ITER}")
    if not os.path.exists(new_pc):
        shutil.copytree(old_pc, new_pc)

    # Load config overrides
    with open(os.path.join(PROJECT_DIR, "configs", "scaffold_resume.yaml")) as f:
        cfg = yaml.safe_load(f)

    # Build parsers and get ALL defaults, then override
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # Base defaults from empty parse
    opt_overrides = {
        "iterations": cfg["iterations"],
        "position_lr_init": cfg["position_lr_init"],
        "position_lr_final": cfg["position_lr_final"],
        "position_lr_max_steps": cfg["position_lr_max_steps"],
        "offset_lr_init": cfg["offset_lr_init"],
        "offset_lr_final": cfg["offset_lr_final"],
        "offset_lr_max_steps": cfg["offset_lr_max_steps"],
        "feature_lr": cfg["feature_lr"], "opacity_lr": cfg["opacity_lr"],
        "scaling_lr": cfg["scaling_lr"], "rotation_lr": cfg["rotation_lr"],
        "mlp_opacity_lr_init": cfg["mlp_opacity_lr_init"],
        "mlp_opacity_lr_final": cfg["mlp_opacity_lr_final"],
        "mlp_opacity_lr_max_steps": cfg["mlp_opacity_lr_max_steps"],
        "mlp_cov_lr_init": cfg["mlp_cov_lr_init"],
        "mlp_cov_lr_final": cfg["mlp_cov_lr_final"],
        "mlp_cov_lr_max_steps": cfg["mlp_cov_lr_max_steps"],
        "mlp_color_lr_init": cfg["mlp_color_lr_init"],
        "mlp_color_lr_final": cfg["mlp_color_lr_final"],
        "mlp_color_lr_max_steps": cfg["mlp_color_lr_max_steps"],
        "percent_dense": cfg["percent_dense"], "lambda_dssim": cfg["lambda_dssim"],
        "start_stat": cfg["start_stat"], "update_from": cfg["update_from"],
        "update_interval": cfg["update_interval"], "update_until": cfg["update_until"],
        "min_opacity": cfg["min_opacity"], "success_threshold": cfg["success_threshold"],
        "densify_grad_threshold": cfg["densify_grad_threshold"],
    }
    opt_args = get_default_opt(parser, opt_overrides)

    # ModelParams defaults + overrides
    model_overrides = {
        "source_path": os.path.join(scene_data, "train"),
        "model_path": new_out,
        "images": "images",
        "resolution": 1,
        "white_background": False,
        "data_device": "cuda",
        "eval": True,
        "feat_dim": cfg["feat_dim"], "n_offsets": cfg["n_offsets"],
        "voxel_size": cfg["voxel_size"], "update_depth": cfg["update_depth"],
        "update_init_factor": cfg["update_init_factor"],
        "update_hierachy_factor": cfg["update_hierachy_factor"],
        "appearance_dim": cfg["appearance_dim"], "ratio": cfg["ratio"],
    }
    model_args = get_default_opt(parser, model_overrides)

    # PipelineParams defaults
    pipe_overrides = {
        "convert_SHs_python": False,
        "compute_cov3D_python": False,
        "debug": False,
    }
    pipe_args = get_default_opt(parser, pipe_overrides)

    dataset = lp.extract(model_args)
    opt = op.extract(opt_args)
    pipe = pp.extract(pipe_args)

    anchor_log = AnchorLogger(os.path.join(new_out, "anchor_log.csv"))
    loss_mon = LossMonitor()
    start_iter = find_latest_iter(new_out, LOAD_ITER) or LOAD_ITER
    print(f"[RESUME] {scene_name}: start from iteration {start_iter}")
    try:
        training_finetune(dataset, opt, pipe, anchor_log, loss_mon, start_iter)
        print(f"\n[SUCCESS] {scene_name}")
        return True
    except Exception as e:
        print(f"\n[FAIL] {scene_name}: {e}")
        import traceback; traceback.print_exc()
        return False


def run_all():
    scenes = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))
    for s in scenes:
        if not run_scene(s): break


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--test_1_scene", type=str, default=None)
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    if a.test_1_scene:
        run_scene(a.test_1_scene)
    elif a.all:
        run_all()
    else:
        print("Usage: --test_1_scene HCM0421  or  --all")
