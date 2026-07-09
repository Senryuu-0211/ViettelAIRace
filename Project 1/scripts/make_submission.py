import os
import shutil
import zipfile
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    submission_dir = os.path.join(args.output_dir, "submission_temp")
    os.makedirs(submission_dir, exist_ok=True)

    # Find all scenes in output_dir
    scenes = [d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d)) and d != "submission_temp"]

    # We need to map scene names to scene_001, scene_002... format
    # The README says:
    # submission_round1.zip
    # ├── scene_001/
    # │   ├── 0001.png
    # ...
    # Let's just copy the folders over for now, maybe mapping them if needed. 
    # Actually, the organizers' dataset is hcm0031, HCM0249. If they want exact folder names to match the input, we keep it. 
    # If they want scene_001, we might need a mapping. The README example uses scene_001.
    # For safety, let's keep the original scene name (e.g. hcm0031) as they must map it back to their GT.

    for scene in scenes:
        renders_dir = os.path.join(args.output_dir, scene, "renders")
        if os.path.exists(renders_dir):
            dest_scene_dir = os.path.join(submission_dir, scene)
            os.makedirs(dest_scene_dir, exist_ok=True)
            for file in os.listdir(renders_dir):
                if file.lower().endswith('.png'):
                    shutil.copy2(os.path.join(renders_dir, file), os.path.join(dest_scene_dir, file))
                    
    zip_path = os.path.join(args.output_dir, "submission_round1.zip")
    print(f"Creating zip file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, submission_dir)
                zipf.write(abs_path, arcname=rel_path)
                
    shutil.rmtree(submission_dir)
    print("Submission zip created successfully.")

if __name__ == "__main__":
    main()
