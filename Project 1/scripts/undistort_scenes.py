"""
Undistort SIMPLE_RADIAL camera images → SIMPLE_PINHOLE for 3DGS compatibility.
Creates a new dataset dir: VAI_NVS_DATA_ROUND2_UNDIST
"""
import os, sys, shutil, struct, cv2, numpy as np

DATA_SRC = r"D:\ViettelAIRace\Project 1\VAI_NVS_DATA_ROUND2"
DATA_DST = r"D:\ViettelAIRace\Project 1\VAI_NVS_DATA_ROUND2_UNDIST"

CAMERA_MODEL_IDS = {0:3, 1:4, 2:4, 3:5, 4:8}
SIMPLE_PINHOLE = 0  # model_id, 3 params
SIMPLE_RADIAL = 2   # model_id, 4 params

for scene in sorted(os.listdir(DATA_SRC)):
    src_cameras = os.path.join(DATA_SRC, scene, "train", "sparse", "0", "cameras.bin")
    if not os.path.exists(src_cameras):
        print(f"[{scene}] SKIP: no cameras.bin")
        continue

    # Read camera model
    with open(src_cameras, "rb") as f:
        n_cams = struct.unpack("<Q", f.read(8))[0]
        cam_data = []
        for i in range(n_cams):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            n_p = CAMERA_MODEL_IDS.get(model_id, 0)
            params = list(struct.unpack("<" + "d" * n_p, f.read(8 * n_p)))
            cam_data.append((cam_id, model_id, w, h, params))
            if model_id != SIMPLE_RADIAL:
                print(f"[{scene}] Camera model={model_id} (not SIMPLE_RADIAL), SKIP")
                break
        else:
            if all(mid == SIMPLE_RADIAL for _, mid, _, _, _ in cam_data):
                f_param = cam_data[0][4][0]
                k_param = cam_data[0][4][3]
                print(f"[{scene}] SIMPLE_RADIAL: f={f_param:.1f} k={k_param:.6f} → converting to PINHOLE")
                
                # Create destination dirs
                dst_scene = os.path.join(DATA_DST, scene)
                os.makedirs(dst_scene, exist_ok=True)
                
                # Copy test/ dir (no changes needed)
                src_test = os.path.join(DATA_SRC, scene, "test")
                dst_test = os.path.join(dst_scene, "test")
                if os.path.exists(src_test) and not os.path.exists(dst_test):
                    shutil.copytree(src_test, dst_test)
                
                # Create train/ dirs
                dst_images = os.path.join(dst_scene, "train", "images")
                dst_sparse = os.path.join(dst_scene, "train", "sparse", "0")
                os.makedirs(dst_images, exist_ok=True)
                os.makedirs(dst_sparse, exist_ok=True)
                
                # Undistort and copy images
                src_images = os.path.join(DATA_SRC, scene, "train", "images")
                img_files = sorted(os.listdir(src_images))
                
                K = np.array([[f_param, 0, w], [0, f_param, h], [0, 0, 1]], dtype=np.float64)
                K[0, 2] = w  # cx
                K[1, 2] = h  # cy
                # Actually: cx and cy from params
                K = np.array([[f_param, 0, cam_data[0][4][1]], 
                              [0, f_param, cam_data[0][4][2]], 
                              [0, 0, 1]], dtype=np.float64)
                dist = np.array([k_param, 0, 0, 0, 0], dtype=np.float64)
                
                for img_file in img_files:
                    src_img = os.path.join(src_images, img_file)
                    dst_img = os.path.join(dst_images, img_file)
                    img = cv2.imread(src_img)
                    if img is None:
                        print(f"  WARN: cannot read {img_file}")
                        shutil.copy2(src_img, dst_img)
                        continue
                    undist = cv2.undistort(img, K, dist, None, K)
                    cv2.imwrite(dst_img, undist)
                
                print(f"  Undistorted {len(img_files)} images")
                
                # Write new cameras.bin with SIMPLE_PINHOLE
                dst_cameras = os.path.join(dst_sparse, "cameras.bin")
                with open(dst_cameras, "wb") as f:
                    f.write(struct.pack("<Q", n_cams))
                    for cam_id, model_id, cw, ch, params in cam_data:
                        # SIMPLE_RADIAL: params = [f, cx, cy, k]
                        # SIMPLE_PINHOLE: params = [f, cx, cy]
                        new_params = params[:3]  # drop k
                        f.write(struct.pack("<iiQQ", cam_id, SIMPLE_PINHOLE, cw, ch))
                        f.write(struct.pack("<" + "d" * 3, *new_params))
                
                # Copy images.bin and points3D.bin (SfM data, unchanged)
                for bin_file in ["images.bin", "points3D.bin"]:
                    src_bin = os.path.join(DATA_SRC, scene, "train", "sparse", "0", bin_file)
                    if os.path.exists(src_bin):
                        shutil.copy2(src_bin, os.path.join(dst_sparse, bin_file))
                
                print(f"  Done → {dst_scene}")

print("\nDone. Undistorted data in:", DATA_DST)
