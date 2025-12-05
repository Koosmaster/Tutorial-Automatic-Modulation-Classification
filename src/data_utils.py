# src/data_utils.py

from typing import Dict, Tuple, List
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os, sys
import json
from pathlib import Path
REPO_URL = "https://github.com/Koosmaster/Tutorial-Automatic-Modulation-Classification.git"
REPO_DIR = Path("/content/Tutorial-Automatic-Modulation-Classification")
TRADITIONAL_DIR = REPO_DIR / "data" / "traditional_models"
def dict_to_iq_arrays(
    radioml_data: Dict[Tuple[str, int], np.ndarray]
):
    """
    Convert a filtered RadioML dictionary into arrays:

      X        : I/Q samples, shape (N, 2, T)
      y        : modulation labels (strings), shape (N,)
      snr_list : SNR values, shape (N,)

    radioml_data is expected to be a dict with keys (mod, snr) and
    values of shape (num_examples, 2, T) or (num_examples, T) complex.
    """
    X_list, y_list, snr_list = [], [], []

    for (mod, snr), signals in radioml_data.items():
        for sig in signals:
            # If sig is complex 1D (T,), turn into (2, T)
            if np.iscomplexobj(sig) and sig.ndim == 1:
                iq = np.vstack((sig.real, sig.imag))
            else:
                iq = sig  # already (2, T)

            X_list.append(iq)
            y_list.append(mod)
            snr_list.append(snr)

    X = np.stack(X_list)        # (N, 2, T)
    y = np.array(y_list)        # (N,)
    snr_arr = np.array(snr_list)

    return X, y, snr_arr


def make_tensor_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = False,
):
    """
    Wrap (N, 2, T) numpy arrays and labels into a PyTorch DataLoader.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    # If labels are strings, convert to ints before calling this function
    y_t = torch.tensor(y, dtype=torch.long)

    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader
    
def load_model_metrics(model_name: str):
    """Load metrics.json for a given model (knn, svm, rf)."""
    model_dir = TRADITIONAL_DIR / model_name
    metrics_path = model_dir / "metrics.json"
    with open(metrics_path, "r") as f:
        return json.load(f)

def show_confusion(model_name: str):
    """Display confusion.png for a given model."""
    img_path = TRADITIONAL_DIR / model_name / "confusion.png"
    display(Image(filename=str(img_path)))
