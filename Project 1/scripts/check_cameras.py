import struct, os, glob

CAMERA_MODEL_IDS = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
    4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV", 7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE"
}
N_PARAMS = {0:3, 1:4, 2:4, 3:5, 4:8, 5:8, 6:12, 7:5, 8:4, 9:5, 10:12}

SUPPORTED = {"SIMPLE_PINHOLE", "PINHOLE"}
DATA_DIR = r"D:\ViettelAIRace\Project 1\VAI_NVS_DATA_ROUND2"

for scene in sorted(os.listdir(DATA_DIR)):
    cams_bin = os.path.join(DATA_DIR, scene, "train", "sparse", "0", "cameras.bin")
    if not os.path.exists(cams_bin):
        print(f"[{scene}] NO cameras.bin")
        continue

    with open(cams_bin, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        counts = {}
        total_w, total_h = 0, 0
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            name = CAMERA_MODEL_IDS.get(model_id, f"UNKNOWN({model_id})")
            counts[name] = counts.get(name, 0) + 1
            n_p = N_PARAMS.get(model_id, 0)
            f.read(8 * n_p)
            total_w = max(total_w, w)
            total_h = max(total_h, h)

    ok = all(m in SUPPORTED for m in counts)
    status = "OK" if ok else "INCOMPATIBLE"
    print(f"[{scene}] {counts}  max={total_w}x{total_h}  {status}")

# Also check round 1 data
print("\n--- Round 1 ---")
DATA_DIR1 = r"D:\ViettelAIRace\Project 1\VAI_NVS_DATA\phase1"
for subset in ["public_set", "private_set1"]:
    subset_dir = os.path.join(DATA_DIR1, subset)
    if not os.path.exists(subset_dir):
        continue
    for scene in sorted(os.listdir(subset_dir)):
        cams_bin = os.path.join(subset_dir, scene, "train", "sparse", "0", "cameras.bin")
        if not os.path.exists(cams_bin):
            print(f"[{subset}/{scene}] NO cameras.bin")
            continue
        with open(cams_bin, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            counts = {}
            for _ in range(n):
                cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
                name = CAMERA_MODEL_IDS.get(model_id, f"UNKNOWN({model_id})")
                counts[name] = counts.get(name, 0) + 1
                n_p = N_PARAMS.get(model_id, 0)
                f.read(8 * n_p)
            ok = all(m in SUPPORTED for m in counts)
            status = "OK" if ok else "INCOMPATIBLE"
            print(f"[{subset}/{scene}] {counts}  {status}")
