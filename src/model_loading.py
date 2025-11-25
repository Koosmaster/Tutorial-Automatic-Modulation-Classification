import os
import zipfile
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_repo_root():
    # <repo_root>/src/model_loading.py → go up 1 level
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_models_unzipped():
    repo_root = _get_repo_root()
    models_dir = os.path.join(repo_root, "models")
    zip_path = os.path.join(models_dir, "models1d_ablation_models.zip")

    # Extract if no .pth files yet
    if os.path.exists(zip_path):
        if not any(f.endswith(".pth") for f in os.listdir(models_dir)):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(models_dir)
                print("Extracted:", zf.namelist())
    return models_dir


def load_ablation_model(config_name, model_cls, best=False, num_classes=11):
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
