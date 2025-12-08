#!/usr/bin/env python3
import argparse
import os
import sys
import time
from datetime import datetime
from statistics import mean, pstdev

# --- Configuration / Steps ---
STEPS = [
    "Data Loading",
    "Data Preprocessing",
    "Data Augmentation",
    "Model Building",
    "Model Training",
    "Validation & Evaluation",
    "Model Saving",
    "Inference Pipeline (demo)",
    "Database Lookup",
    "Response to User"
]

# --- Default feature names (human-friendly) ---
DEFAULT_FEATURES = [
    "aspect_ratio",
    "area_pixels",
    "mean_R",
    "mean_G",
    "mean_B",
    "std_R",
    "std_G",
    "std_B",
    "grayscale_mean",
    "grayscale_std",
    "edge_strength_mean"
]

# --- Utility helpers ---

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def print_header(title):
    print("=" * 80)
    print(f"{title}".center(80))
    print("=" * 80)


def simple_progress(step_name, seconds=1.2, width=36):
    print(f"\n[{timestamp()}] ▶ {step_name}")
    for i in range(width + 1):
        pct = int((i / width) * 100)
        bar = "#" * i + "-" * (width - i)
        print(f"\r    [{bar}] {pct:3d}% ", end="", flush=True)
        time.sleep(seconds / width)
    print("\n    Completed ✅")


# --- Demo/simulated numeric feature generator ---

def simulated_feature_values():
    # simple deterministic simulated values for reproducibility
    vals = [1.25, 10240, 120.4, 98.7, 85.1, 12.3, 10.2, 9.7, 101.6, 8.9, 3.4]
    return vals


# --- Try optional libs for real extraction ---
HAS_PIL = False
HAS_NUMPY = False
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False


# --- Demo placeholders (safe, fast) ---

def demo_load_data(root):
    count = 0
    first_image = None
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                count += 1
                if first_image is None:
                    first_image = os.path.join(dirpath, f)
    print(f"    Found {count} image files under: {root}")
    if first_image:
        print(f"    Example image: {first_image}")
    else:
        print("    No images found in dataset root.")
    time.sleep(0.2)
    return {"image_count": count, "example_image": first_image}


def demo_preprocess(dataset_or_root=None, example_image=None, user_feats=None):
    time.sleep(0.2)
    print("    Preprocessing simulated: resized, normalized, cleaned")
    # Show features and numeric values
    if user_feats and isinstance(user_feats, dict) and user_feats.get('names') and user_feats.get('values'):
        names = user_feats['names']
        values = user_feats['values']
        print("    Features fed to CNN (from project):")
        for n, v in zip(names, values):
            print(f"      - {n}: {v}")
        return True

    # attempt to extract from example image if present
    if example_image:
        feats = extract_features_from_image(example_image)
        if feats:
            print("    Features fed to CNN (extracted from example image):")
            for n, v in zip(DEFAULT_FEATURES[:len(feats)], feats):
                print(f"      - {n}: {v}")
            return True

    # fallback simulated
    vals = simulated_feature_values()
    print("    Features fed to CNN (simulated):")
    for n, v in zip(DEFAULT_FEATURES, vals):
        print(f"      - {n}: {v}")
    return True

def demo_augment():
    time.sleep(0.15)
    print("    Augmentation simulated: rotations, flips, brightness jitter")
    return True

def demo_build_model():
    time.sleep(0.15)
    print("    Model built (demo architecture: Conv->Pool->Dense)")
    return True


def demo_train(epochs=2):
    for e in range(1, epochs + 1):
        print(f"    Epoch {e}/{epochs} ...", end="", flush=True)
        for _ in range(3):
            print(".", end="", flush=True)
            time.sleep(0.25)
        print(" done")
    time.sleep(0.1)
    return {"epochs": epochs, "final_loss": 0.12, "accuracy": 0.93}

def demo_validate():
    time.sleep(0.12)
    print("    Validation: accuracy=90.8%  loss=0.28")
    return {"val_acc": 0.908}


def demo_save_model():
    time.sleep(0.08)
    print("    Model saved to: models/demo_model.h5")
    return "models/demo_model.h5"


def demo_inference(image_path=None):
    time.sleep(0.1)
    print(f"    Running inference on: {image_path or 'demo_image.jpg'}")
    return ("Ocimum sanctum", 0.923)


def demo_lookup(label):
    time.sleep(0.06)
    return {
        "name": label,
        "uses": "anti-inflammatory, antioxidant, respiratory benefits",
        "confidence_note": "demo data"
    }


# --- Feature extraction from image (uses PIL and numpy if available) ---

def extract_features_from_image(path):
    if not os.path.exists(path):
        return None
    if not HAS_PIL or not HAS_NUMPY:
        # cannot do real extraction without PIL+numpy
        return None
    try:
        img = Image.open(path).convert('RGB')
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        area = h * w
        aspect = round(w / h, 3) if h != 0 else 0.0
        mean_rgb = arr.mean(axis=(0,1))
        std_rgb = arr.std(axis=(0,1))
        # grayscale for simple stats
        gray = arr.mean(axis=2)
        gray_mean = float(round(gray.mean(), 3))
        gray_std = float(round(gray.std(), 3))
        # very simple edge strength: mean of gradient magnitude
        gy, gx = np.gradient(gray)
        grad = np.sqrt(gx**2 + gy**2)
        edge_strength_mean = float(round(grad.mean(), 3))
        features = [
            round(float(aspect), 3),
            int(area),
            float(round(mean_rgb[0], 3)),
            float(round(mean_rgb[1], 3)),
            float(round(mean_rgb[2], 3)),
            float(round(std_rgb[0], 3)),
            float(round(std_rgb[1], 3)),
            float(round(std_rgb[2], 3)),
            gray_mean,
            gray_std,
            edge_strength_mean
        ]
        return features
    except Exception:
        return None


# --- Attempt to import user functions from src/ (non-fatal) ---
USER_FUNCS = {
    "load_data": None,
    "preprocess_data": None,
    "augment_data": None,
    "build_model": None,
    "train_model": None,
    "validate": None,
    "save_model": None,
    "run_inference": None,
    "lookup_medicinal_info": None,
    "get_features": None,  # optional: user-provided list of feature names
}


def try_imports():
    try:
        from src.data.loader import load_data as user_load
        USER_FUNCS['load_data'] = user_load
        print("[import] Found: src.data.loader.load_data")
    except Exception:
        pass

    try:
        from src.preprocess import preprocess_data as user_pre
        USER_FUNCS['preprocess_data'] = user_pre
        print("[import] Found: src.preprocess.preprocess_data")
    except Exception:
        pass

    try:
        from src.augment import augment_data as user_aug
        USER_FUNCS['augment_data'] = user_aug
        print("[import] Found: src.augment.augment_data")
    except Exception:
        pass

    try:
        from src.model import build_model as user_build
        USER_FUNCS['build_model'] = user_build
        print("[import] Found: src.model.build_model")
    except Exception:
        pass

    try:
        from src.training import train_model as user_train
        USER_FUNCS['train_model'] = user_train
        print("[import] Found: src.training.train_model")
    except Exception:
        pass

    try:
        from src.validation import validate as user_val
        USER_FUNCS['validate'] = user_val
        print("[import] Found: src.validation.validate")
    except Exception:
        pass

    try:
        from src.save import save_model as user_save
        USER_FUNCS['save_model'] = user_save
        print("[import] Found: src.save.save_model")
    except Exception:
        pass

    try:
        from src.inference import run_inference as user_infer
        USER_FUNCS['run_inference'] = user_infer
        print("[import] Found: src.inference.run_inference")
    except Exception:
        pass

    try:
        from src.database import lookup_medicinal_info as user_lookup
        USER_FUNCS['lookup_medicinal_info'] = user_lookup
        print("[import] Found: src.database.lookup_medicinal_info")
    except Exception:
        pass

    try:
        from src.features import get_features as user_feats
        USER_FUNCS['get_features'] = user_feats
        print("[import] Found: src.features.get_features")
    except Exception:
        pass


# --- Runner wrapper: uses real func if provided else demo ---

def run_step(step_name, real_func=None, *args, **kwargs):
    print(f"\n[{timestamp()}] ▶ {step_name}")

    # If a real function exists, try calling it first (non-fatal)
    if real_func:
        try:
            res = real_func(*args, **kwargs)
            print("    Completed ✅")
            return res
        except Exception as e:
            print(f"    Failed ❌  Error while running real function: {e}")
            print("    Falling back to demo for this step.")

    # Demo fallbacks and special handling
    if step_name == "Data Loading":
        return demo_load_data(kwargs.get('root') or (args[0] if args else '.'))

    if step_name == "Data Preprocessing":
        # Priority order to show real features/values:
        # 1) If user provided src.features.get_features(), call it and try to get numeric values
        user_feature_obj = None
        if USER_FUNCS.get('get_features'):
            try:
                names = USER_FUNCS['get_features']()
                # if user's preprocess_data returns numeric vector, attempt that
                if USER_FUNCS.get('preprocess_data'):
                    try:
                        prep_res = USER_FUNCS['preprocess_data'](kwargs.get('root'))
                        # Expecting a dict like {'names': [...], 'values': [...]}
                        if isinstance(prep_res, dict) and prep_res.get('values'):
                            user_feature_obj = {'names': prep_res.get('names', names), 'values': prep_res.get('values')}
                    except Exception:
                        # ignore failure to call preprocess
                        user_feature_obj = {'names': names, 'values': None}
                else:
                    user_feature_obj = {'names': names, 'values': None}
                # print names
                print("    Features fed to CNN (from src.features.get_features()):")
                for n in names:
                    print(f"      - {n}")
            except Exception as e:
                print(f"    Could not call src.features.get_features(): {e}")
                user_feature_obj = None

        # 2) If user's preprocess_data returned numeric values inside user_feature_obj, show them
        if user_feature_obj and user_feature_obj.get('values'):
            print("    Numeric feature values (from project's preprocess_data):")
            for n, v in zip(user_feature_obj.get('names', DEFAULT_FEATURES), user_feature_obj.get('values')):
                print(f"      - {n}: {v}")
            return True

        # 3) Try to extract from first image under root using PIL+numpy
        example_image = kwargs.get('example_image')
        if not example_image and isinstance(kwargs.get('root'), str):
            # find first image under root
            root = kwargs.get('root')
            for dirpath, _, filenames in os.walk(root):
                for f in filenames:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        example_image = os.path.join(dirpath, f)
                        break
                if example_image:
                    break

        if example_image:
            feats = extract_features_from_image(example_image)
            if feats:
                print("    Features fed to CNN (extracted from example image):")
                for n, v in zip(DEFAULT_FEATURES, feats):
                    print(f"      - {n}: {v}")
                return True

        # 4) Fallback to simulated values
        return demo_preprocess(dataset_or_root=kwargs.get('root'), example_image=example_image, user_feats=None)

    if step_name == "Data Augmentation":
        return demo_augment()

    if step_name == "Model Building":
        return demo_build_model()

    if step_name == "Model Training":
        return demo_train(kwargs.get('epochs', 2))

    if step_name == "Validation & Evaluation":
        return demo_validate()

    if step_name == "Model Saving":
        return demo_save_model()

    if step_name.startswith("Inference"):
        return demo_inference(kwargs.get('image_path'))

    if step_name == "Database Lookup":
        return demo_lookup(kwargs.get('label', 'Unknown'))

    if step_name == "Response to User":
        print("    Predicted: Ocimum sanctum (Tulsi)  Confidence: 92.3%")
        print("    Medicinal uses: anti-inflammatory, antioxidant, respiratory benefits")
        return True

    # final fallback
    simple_progress(step_name)
    return None


# --- Main flow orchestration ---
def main():
    parser = argparse.ArgumentParser(description="Show project flow in terminal (real or demo).")
    parser.add_argument("--data-root", "-d", default="..", help="Path to dataset root (defaults to parent dir).")
    parser.add_argument("--simulate-only", action="store_true", help="Force demo-only mode (ignore real imports).")
    parser.add_argument("--use-real", action="store_true", help="Attempt to import and use functions from src/ (non-fatal).")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for training/demo")
    parser.add_argument("--step", type=str, default=None, help="If set, only run this single step (name must match).")
    args = parser.parse_args()

    print_header("Medicinal Plant Recognition — Project Flow (Terminal Demo)")
    print(f"Started at: {timestamp()}")

    if args.use_real and not args.simulate_only:
        try_imports()
    else:
        if args.use_real and args.simulate_only:
            print("[note] --simulate-only given; skipping real imports even though --use-real was set.")

    # map steps to possible user functions
    step_to_func = {
        "Data Loading": USER_FUNCS.get('load_data'),
        "Data Preprocessing": USER_FUNCS.get('preprocess_data'),
        "Data Augmentation": USER_FUNCS.get('augment_data'),
        "Model Building": USER_FUNCS.get('build_model'),
        "Model Training": USER_FUNCS.get('train_model'),
        "Validation & Evaluation": USER_FUNCS.get('validate'),
        "Model Saving": USER_FUNCS.get('save_model'),
        "Inference Pipeline (demo)": USER_FUNCS.get('run_inference'),
        "Database Lookup": USER_FUNCS.get('lookup_medicinal_info'),
        "Response to User": None,
    }

    # helper to run single step if requested
    def run_single(step_name):
        real = None if args.simulate_only else step_to_func.get(step_name)
        # Pass sensible kwargs for common functions
        kwargs = {}
        if step_name == "Data Loading":
            kwargs['root'] = args.data_root
        if step_name == "Model Training":
            kwargs['epochs'] = args.epochs
        if step_name.startswith("Inference"):
            kwargs['image_path'] = os.path.join(args.data_root, 'demo_image.jpg')
        if step_name == "Database Lookup":
            kwargs['label'] = 'Ocimum sanctum'
        return run_step(step_name, real, **kwargs)

    # If user asked to run a single step
    if args.step:
        target = args.step.strip()
        if target not in STEPS and target != "Inference Pipeline (demo)":
            print(f"[error] Unknown step: {target}")
            print("Available steps:")
            for s in STEPS:
                print("  - ", s)
            sys.exit(1)
        run_single(target)
        print("\nDone (single-step).")
        return

    # Full flow
    data = run_single(STEPS[0])
    _ = run_single(STEPS[1])
    _ = run_single(STEPS[2])
    _ = run_single(STEPS[3])
    train_res = run_single(STEPS[4])
    _ = run_single(STEPS[5])
    model_path = run_single(STEPS[6])
    infer_res = run_single(STEPS[7])

    # inference -> label -> lookup
    if isinstance(infer_res, tuple) and len(infer_res) >= 1:
        label = infer_res[0]
    else:
        label = 'Ocimum sanctum'
    lookup_res = run_step("Database Lookup", USER_FUNCS.get('lookup_medicinal_info'), label=label)

    # final response
    run_step("Response to User")

    print("\nAll steps completed. Presentation finished ✅")
    print("=" * 80)
    print(f"Finished at: {timestamp()}")


if __name__ == '__main__':
    main()
