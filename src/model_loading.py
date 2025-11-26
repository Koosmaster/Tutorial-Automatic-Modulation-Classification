# src/model_loading.py
import os, zipfile, torch

device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR   = os.path.dirname(__file__)           # .../src
MODELS_DIR = os.path.join(BASE_DIR, "models")    # .../src/models
ZIP_PATH   = os.path.join(MODELS_DIR, "models1d_ablation_models.zip")

def _ensure_unzipped():
    if os.path.exists(ZIP_PATH) and not any(f.endswith(".pth") for f in os.listdir(MODELS_DIR)):
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(MODELS_DIR)

def load_ablation_model(config_name, model_cls, best: bool = False, num_classes: int = 11):
    _ensure_unzipped()

    fname = f"best_{config_name}.pth" if best else f"{config_name}.pth"
    ckpt_path = os.path.join(MODELS_DIR, fname)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = model_cls(num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
    
# =========================================================
# LOAD TRANSFORMER MODEL
# =========================================================

def load_transformer_model(model_cls, fname="transformer_raw_iq.pth", num_classes=11):
    """
    Load a saved Raw I/Q Transformer model.
    """
    _ensure_unzipped()  # safe to call even if no zip is used

    ckpt_path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Transformer checkpoint not found: {ckpt_path}")

    model = model_cls(num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
