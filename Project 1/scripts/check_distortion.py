import struct, os

CAMERA_MODEL_IDS = {0:3, 1:4, 2:4, 3:5, 4:8, 5:8, 6:12, 7:5, 8:4, 9:5, 10:12}
DATA_DIR = r"D:\ViettelAIRace\Project 1\VAI_NVS_DATA_ROUND2"

for scene in sorted(os.listdir(DATA_DIR)):
    cams_bin = os.path.join(DATA_DIR, scene, "train", "sparse", "0", "cameras.bin")
    if not os.path.exists(cams_bin):
        continue
    with open(cams_bin, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for i in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            n_p = CAMERA_MODEL_IDS.get(model_id, 0)
            params = struct.unpack("<" + "d" * n_p, f.read(8 * n_p))
            if model_id == 2:  # SIMPLE_RADIAL
                fparam, cx, cy, k = params
                print(f"{scene}: f={fparam:.1f} cx={cx:.1f} cy={cy:.1f} k={k:.6f}  img={w}x{h}")
            elif model_id == 0:  # SIMPLE_PINHOLE
                fparam, cx, cy = params
                print(f"{scene}: f={fparam:.1f} cx={cx:.1f} cy={cy:.1f}  img={w}x{h}")
            elif model_id == 1:  # PINHOLE
                fx, fy, cx, cy = params
                print(f"{scene}: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  img={w}x{h}")
            else:
                print(f"{scene}: model_id={model_id} params={params}  img={w}x{h}")
