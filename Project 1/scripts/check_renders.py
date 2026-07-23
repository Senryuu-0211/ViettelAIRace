from PIL import Image
import numpy as np, os

base = r"D:\ViettelAIRace\Project 1"
scene = "HCM0674"

for name, subdir in [("RENDER", "output_round2"), ("TRAIN", "VAI_NVS_DATA_ROUND2")]:
    img_dir = os.path.join(base, subdir, scene, "train" if subdir=="VAI_NVS_DATA_ROUND2" else "", "images" if subdir=="VAI_NVS_DATA_ROUND2" else "renders")
    if not os.path.exists(img_dir):
        alt = os.path.join(base, subdir, scene, "renders")
        if os.path.exists(alt): img_dir = alt
    imgs = sorted(os.listdir(img_dir))[:3]
    print(f"--- {name} ({img_dir}) ---")
    for f in imgs:
        img = np.array(Image.open(os.path.join(img_dir, f)).convert("RGB"))
        print(f"  {f}: mean={img.mean():.0f} min={img.min()} max={img.max()} shape={img.shape}")
