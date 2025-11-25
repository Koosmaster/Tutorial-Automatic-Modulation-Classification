
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_ablation_model(config_name, model_cls, best=False,
                        num_classes=11, repo_root=None):
    """
    Load a pretrained 1D CNN ablation configuration checkpoint.

    Args:
        config_name (str): e.g. "T_stepLR"
        model_cls (nn.Module class): model class to instantiate, e.g. AMC1DCNN_v2
        best (bool): if True, load best_<config>.pth instead of <config>.pth
        num_classes (int): number of modulation classes
        repo_root (str or None): repo root; defaults to current working directory

    Returns:
        model (nn.Module): loaded model in eval mode on the correct device
    """
    if repo_root is None:
        repo_root = os.getcwd()

    models_dir = os.path.join(repo_root, "src", "models")
    fname = f"best_{config_name}.pth" if best else f"{config_name}.pth"
    ckpt_path = os.path.join(models_dir, fname)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Instantiate model from the class passed in
    model = model_cls(num_classes=num_classes).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
