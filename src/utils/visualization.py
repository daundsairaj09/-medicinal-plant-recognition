# src/utils/visualization.py

import os
import matplotlib.pyplot as plt

def plot_training_curves(history, out_dir, prefix="model"):
    """
    Save training & validation accuracy/loss curves as PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    # Accuracy plot
    plt.figure()
    plt.plot(acc, label="train_accuracy")
    plt.plot(val_acc, label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.savefig(os.path.join(out_dir, f"{prefix}_accuracy.png"))
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(out_dir, f"{prefix}_loss.png"))
    plt.close()
