# src/plot_utils.py

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Sequence
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_history_curves(history: Dict[str, Sequence[float]], title: str = ""):
    """
    Plot train/val loss and accuracy from a history dict with keys:
      'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} – Loss")
    plt.grid(True)
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} – Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_snr_curve(snrs, accs, label: str = ""):
    """
    Plot accuracy vs SNR.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(snrs, accs, marker="o", label=label or "model")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs SNR {f'– {label}' if label else ''}")
    plt.grid(True)
    if label:
        plt.legend()
    plt.show()


def plot_confusion(y_true, y_pred, class_names=None, title: str = ""):
    """
    Compute and plot a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names if class_names is not None else "auto",
        yticklabels=class_names if class_names is not None else "auto",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title or "Confusion Matrix")
    plt.tight_layout()
    plt.show()
