# src/eval_utils.py

from typing import Tuple, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    """
    Return (#correct, #total) from raw logits and integer labels.
    """
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct, total


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    criterion=None,
) -> Dict[str, object]:
    """
    Run a full pass over a DataLoader.

    Returns dict with:
      - 'loss'
      - 'acc'
      - 'preds'
      - 'labels'
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += loss.item() * y.size(0)

            correct, cnt = accuracy_from_logits(logits, y)
            total_correct += correct
            total_count += cnt

            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    out = {
        "loss": total_loss / max(1, total_count) if criterion is not None else None,
        "acc": total_correct / max(1, total_count),
        "preds": preds,
        "labels": labels,
    }
    return out


def snr_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
):
    """
    Compute accuracy as a function of SNR.

    Returns (unique_snrs, accuracies).
    """
    unique_snrs = np.unique(snrs)
    accs = []

    for snr in unique_snrs:
        idx = np.where(snrs == snr)[0]
        if len(idx) == 0:
            accs.append(0.0)
        else:
            accs.append((preds[idx] == labels[idx]).mean())

    return unique_snrs, np.array(accs)
