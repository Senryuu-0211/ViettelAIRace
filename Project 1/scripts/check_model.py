import torch, os

for chkpnt in ['chkpnt1000.pth', 'chkpnt3000.pth', 'chkpnt6000.pth']:
    path = os.path.join('output/hcm0031', chkpnt)
    if not os.path.exists(path):
        continue
    model_params, iter_num = torch.load(path, map_location='cpu')
    opacity_raw = model_params[6]
    opacity = torch.sigmoid(opacity_raw)
    n_points = model_params[1].shape[0]
    features_dc = model_params[2]
    print(f"\n=== {chkpnt} (iter {iter_num}, {n_points} points) ===")
    print(f"  Opacity raw: min={opacity_raw.min():.4f}, max={opacity_raw.max():.4f}, mean={opacity_raw.mean():.4f}")
    print(f"  Opacity sig: min={opacity.min():.4f}, max={opacity.max():.4f}, mean={opacity.mean():.4f}")
    print(f"  Features DC: min={features_dc.min():.4f}, max={features_dc.max():.4f}, mean={features_dc.mean():.4f}")
