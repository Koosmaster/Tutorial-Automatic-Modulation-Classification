# src/model_loading.py

import os
import zipfile
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_repo_root():
    """Assume this file lives in <repo_root>/src/."""
    return os.path.dirname(os.path.dirname(__file__))


def _ensure_models_unzipped():
    """
    Make sure the ablation .pth files are extracted from models1d_ablation_models.zip.
    Safe to call multiple times.
    """
    repo_root = _get_repo_root()
    models_dir = os.path.join(repo_root, "src", "models")
    zip_path = os.path.join(models_dir, "models1d_ablation_models.zip")

    # If the zip is missing, just return; caller will error later on missing .pth
    if not os.path.exists(zip_path):
        return models_dir

    # If we already have some .pth files, assume we've extracted
    if any(fname.endswith(".pth") for fname in os.listdir(models_dir)):
        return models_dir

    # Otherwise extract all model weights
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(models_dir)

    return models_dir


def load_ablation_model(config_name, model_cls, best: bool = False,
                        num_classes: int = 11):
    """
    Load a pretrained 1D CNN ablation configuration checkpoint.

    Args:
        config_name (str): e.g. "T_stepLR"
        model_cls (nn.Module class): e.g. AMC1DCNN_v2
        best (bool): if True, loads best_<config_name>.pth
        num_classes (int): number of modulation classes

    Returns:
        model (nn.Module): loaded model in eval mode on the correct device
    """
    models_dir = _ensure_models_unzipped()

    fname = f"best_{config_name}.pth" if best else f"{config_name}.pth"
    ckpt_path = os.path.join(models_dir, fname)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = model_cls(num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
