# src/data/make_splits.py

import os
import shutil
import random
from src.config import RAW_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR

# Train / Val / Test split
SPLIT_RATIO = (0.8, 0.1, 0.1)  # 80/10/10
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")


def split_data():
    # Clean / create processed dirs
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    # Loop over each "plant" folder inside data/raw
    plants = [
        p for p in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, p))
    ]

    print(f"🔎 Found {len(plants)} plant folders in data/raw")

    for plant in plants:
        plant_path = os.path.join(RAW_DATA_DIR, plant)

        # Take only real image files
        images = [
            f for f in os.listdir(plant_path)
            if os.path.isfile(os.path.join(plant_path, f))
            and f.endswith(VALID_EXT)
        ]

        if not images:
            print(f"⚠️ Skipping '{plant}' (no valid image files found)")
            continue

        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * SPLIT_RATIO[0])
        n_val = int(n_total * SPLIT_RATIO[1])
        n_test = n_total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        splits = [
            (TRAIN_DIR, train_imgs),
            (VAL_DIR, val_imgs),
            (TEST_DIR, test_imgs),
        ]

        for split_root, img_list in splits:
            split_plant_dir = os.path.join(split_root, plant)
            os.makedirs(split_plant_dir, exist_ok=True)

            for img in img_list:
                src = os.path.join(plant_path, img)
                dst = os.path.join(split_plant_dir, img)
                shutil.copy(src, dst)

        print(f"✅ Split '{plant}': "
              f"{len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    print("🎉 Dataset split completed successfully.")


if __name__ == "__main__":
    split_data()
