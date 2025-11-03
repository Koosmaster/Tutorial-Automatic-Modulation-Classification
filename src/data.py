## File is called in the Notebooks, this handles loading the dataset and preprocessing
import pickle, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_radioml(path):
    """Load the RadioML 2016.10A dataset pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def to_xy(d):
    """Convert RadioML dict to X, mods, snrs arrays."""
    X, mods, snrs = [], [], []
    for (mod, snr), arr in d.items():
        X.append(arr)
        mods += [mod]*len(arr)
        snrs += [snr]*len(arr)
    X = np.vstack(X).astype(np.float32)
    return X, np.array(mods), np.array(snrs)

def encode_labels(mods):
    """Encode modulation labels into integers."""
    enc = LabelEncoder()
    y = enc.fit_transform(mods)
    return y, list(enc.classes_), enc

def stratified_split(X, y, test_size=0.3, seed=42):
    """Stratified train/validation split."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
