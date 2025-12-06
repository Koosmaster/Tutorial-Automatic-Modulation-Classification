import json
import torch

from models_rnn import AMCRNN   # adjust to your actual module/class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load manifest to get architecture config
with open(manifest_path, "r") as f:
    manifest = json.load(f)

model_cfg = manifest["model_cfg"]  # or whatever key you used

# Rebuild model
model = AMCRNN(
    num_classes=manifest["num_classes"],
    cell_type=model_cfg.get("cell_type", "gru"),
    hidden_size=model_cfg.get("hidden_size", 128),
    num_layers=model_cfg.get("num_layers", 1),
    bidirectional=model_cfg.get("bidirectional", False),
    dropout=model_cfg.get("dropout", 0.0),
).to(device)

# Load checkpoint
ckpt = torch.load(best_ckpt_path, map_location=device)

# Common patterns:
state_dict = (
    ckpt.get("model_state_dict")
    or ckpt.get("model")
    or ckpt        # if you saved just the state_dict
)

model.load_state_dict(state_dict)
model.eval()
