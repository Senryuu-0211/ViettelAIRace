import os
import argparse
import json
import ssl
# Fix SSL certificate issues on Windows for model downloads
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import numpy as np
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim

def calculate_psnr_norm(img1, img2):
    # normalize to [0,1]
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    # Standard PSNR range is ~20 to ~40. The metric asks for PSNR_norm [0, 1].
    # A common normalization is psnr / 100, clamped to [0, 1].
    psnr_norm = min(max(psnr / 100.0, 0.0), 1.0)
    return psnr_norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.gt_dir):
        print(f"Warning: GT directory {args.gt_dir} does not exist. Skipping evaluation.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    total_psnr_norm = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0

    for file in os.listdir(args.pred_dir):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        pred_path = os.path.join(args.pred_dir, file)
        gt_path = os.path.join(args.gt_dir, file)
        
        # In this dataset, GT files might have different extensions (.JPG vs .png)
        # We need to find the matching GT file ignoring extension
        base_name = os.path.splitext(file)[0]
        gt_files = [f for f in os.listdir(args.gt_dir) if os.path.splitext(f)[0] == base_name]
        
        if not gt_files:
            print(f"Warning: No GT found for {file}")
            continue
            
        gt_path = os.path.join(args.gt_dir, gt_files[0])

        pred_img = Image.open(pred_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        
        # Ensure sizes match
        if pred_img.size != gt_img.size:
            pred_img = pred_img.resize(gt_img.size, Image.BILINEAR)

        pred_np = np.array(pred_img)
        gt_np = np.array(gt_img)

        # PSNR norm
        psnr_norm = calculate_psnr_norm(pred_np, gt_np)

        # SSIM
        s = ssim(gt_np, pred_np, channel_axis=2, data_range=255)

        # LPIPS
        pred_tensor = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        gt_tensor = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        
        pred_tensor = pred_tensor.to(device)
        gt_tensor = gt_tensor.to(device)
        
        with torch.no_grad():
            l = loss_fn_vgg(pred_tensor, gt_tensor).item()

        total_psnr_norm += psnr_norm
        total_ssim += s
        total_lpips += l
        count += 1

    if count > 0:
        avg_psnr_norm = total_psnr_norm / count
        avg_ssim = total_ssim / count
        avg_lpips = total_lpips / count
        score = 0.4 * (1 - avg_lpips) + 0.3 * avg_ssim + 0.3 * avg_psnr_norm
        
        metrics = {
            "PSNR_norm": avg_psnr_norm,
            "SSIM": avg_ssim,
            "LPIPS": avg_lpips,
            "Final_Score": score,
            "Count": count
        }
        
        print(f"Evaluation Results: {metrics}")
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
    else:
        print("No matching files found for evaluation.")

if __name__ == "__main__":
    main()
