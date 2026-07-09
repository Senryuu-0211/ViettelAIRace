import os
import struct

import collections

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class Image(BaseImage):
    pass

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            if model_id in CAMERA_MODEL_IDS:
                model_name = CAMERA_MODEL_IDS[model_id].model_name
                num_params = CAMERA_MODEL_IDS[model_id].num_params
            else:
                model_name = "UNKNOWN"
                num_params = 0
            width = camera_properties[2]
            height = camera_properties[3]
            params = struct.unpack("<" + "d" * num_params,
                                   fid.read(8 * num_params))
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=params)
        assert len(cameras) == num_cameras
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack(
                "<idddddddi", fid.read(64))
            image_id = binary_image_properties[0]
            qvec = binary_image_properties[1:5]
            tvec = binary_image_properties[5:8]
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = struct.unpack("<c", fid.read(1))[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("<c", fid.read(1))[0]
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            x_y_id_s = struct.unpack("<" + "ddq" *
                                     num_points2D, fid.read(24 * num_points2D))
            xys = []
            point3D_ids = []
            for i in range(num_points2D):
                xys.append((x_y_id_s[3*i], x_y_id_s[3*i+1]))
                point3D_ids.append(x_y_id_s[3*i+2])
            images[image_id] = Image(
                id=image_id, qvec=list(qvec), tvec=list(tvec),
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def main():
    scene_dir = r"d:\ViettelAIRace\Project 1\VAI_NVS_DATA\phase1\public_set\hcm0031\train\sparse\0"
    
    cameras_file = os.path.join(scene_dir, "cameras.bin")
    images_file = os.path.join(scene_dir, "images.bin")
    
    print("--- SANITY CHECK COLMAP FORMAT ---")
    print(f"Checking directory: {scene_dir}")
    
    # 1. Check cameras.bin
    try:
        print("\nReading cameras.bin...")
        cameras = read_cameras_binary(cameras_file)
        print(f"SUCCESS: Read {len(cameras)} cameras.")
        for cam_id, cam in cameras.items():
            print(f"  Camera {cam_id}: {cam.model} ({cam.width}x{cam.height})")
    except Exception as e:
        print(f"FAILED to read cameras.bin: {e}")
        
    # 2. Check images.bin
    try:
        print("\nReading images.bin...")
        images = read_images_binary(images_file)
        print(f"SUCCESS: Read {len(images)} images.")
        # print first 3 images
        for i, (img_id, img) in enumerate(images.items()):
            if i < 3:
                print(f"  Image {img_id}: {img.name} | Camera ID: {img.camera_id} | Num points: {len(img.point3D_ids)}")
            elif i == 3:
                print("  ...")
    except Exception as e:
        print(f"FAILED to read images.bin: {e}")

if __name__ == "__main__":
    main()
