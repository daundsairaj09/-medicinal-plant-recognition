# src/data/clean_bad_images.py

import os
from PIL import Image
from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR

def clean_split(root):
    removed = 0
    print(f"\n🔍 Checking images in: {root}")
    for folder, _, files in os.walk(root):
        for f in files:
            path = os.path.join(folder, f)
            try:
                with Image.open(path) as img:
                    img.verify()   # validate image
            except Exception as e:
                print(f"❌ Removing corrupted file: {path} | Reason: {e}")
                try:
                    os.remove(path)
                    removed += 1
                except Exception as e2:
                    print(f"   ⚠️ Could not delete {path}: {e2}")
    print(f"✅ Done {root} – removed {removed} bad files")


def main():
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.isdir(split_dir):
            clean_split(split_dir)
        else:
            print(f"⚠️ Directory not found: {split_dir}")


if __name__ == "__main__":
    main()
