# src/utils/metrics.py

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_predictions(y_true, y_pred, class_names):
    """
    Print classification report and confusion matrix.
    """
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    return cm
