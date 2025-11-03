## handles the model training and predicition
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def train_classifier(model, Xtr, ytr, Xva, yva, *, epochs=10, lr=1e-3, bs=256):
    """Train classifier on given data."""
    device = get_device()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    tr_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    va_ds = TensorDataset(torch.tensor(Xva), torch.tensor(yva))
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=bs)

    for _ in range(epochs):
        model.train()
        for xb, yb in tqdm(tr_dl, leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
    return model

def predict(model, X):
    """Run forward pass and return predicted labels."""
    device = get_device()
    model.eval().to(device)
    with torch.no_grad():
        return model(torch.tensor(X).to(device)).argmax(1).cpu().numpy()
