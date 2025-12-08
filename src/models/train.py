# src/models/train.py

import os
import json
import argparse

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from src.config import MODEL_DIR, IMG_SIZE, EPOCHS
from src.data.dataloader import get_data_generators
from src.models.baseline_cnn import build_baseline_cnn
from src.models.mobilenet_model import build_mobilenet_model
from src.utils.visualization import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train medicinal plant classification model")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet",
        choices=["baseline", "mobilenet"],
        help="Which model to train"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs"
    )
    return parser.parse_args()


def main():
    print("🚀 Starting training script")

    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"📁 MODEL_DIR: {MODEL_DIR}")

    # Load data
    print("📦 Loading data generators...")
    train_data, val_data = get_data_generators()
    num_classes = train_data.num_classes
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

    print(f"✅ Found {num_classes} classes")
    print("Class indices:", train_data.class_indices)

    # Save class index mapping
    class_indices_path = os.path.join(MODEL_DIR, "class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(train_data.class_indices, f, indent=2)
    print(f"💾 Saved class indices to {class_indices_path}")

    # Build model
    print(f"🧠 Building model: {args.model}")
    if args.model == "baseline":
        model = build_baseline_cnn(input_shape, num_classes)
        model_name = "baseline_cnn"
    else:
        model = build_mobilenet_model(input_shape, num_classes, train_base=False)
        model_name = "mobilenet_v2"

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    print("🏃‍♂️ Starting training...")
    history = model.fit(
        train_data,
        epochs=args.epochs,
        validation_data=val_data,
        callbacks=callbacks
    )

    # Plot curves
    print("📊 Saving training curves...")
    plot_training_curves(history, out_dir=MODEL_DIR, prefix=model_name)

    print(f"✅ Training finished. Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Training crashed with error:")
        import traceback
        traceback.print_exc()
        raise