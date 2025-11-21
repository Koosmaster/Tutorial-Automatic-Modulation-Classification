"""
data.py — Helper functions for loading the RadioML dataset and managing
train/val/test splits.

Used across all project notebooks.
"""
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Load RadioML 2016.10A Dataset
def load_radioml_pkl_dataset(filepath, filter_analog=False):
    """
    Load the RadioML 2016.10A dataset from a pickle file.
    Optionally filters for analog modulations.
    Returns:
      - radioml_dict: dict with keys (modulation_str, SNR) and values as list of complex signals.
      - unique_mods: list of unique modulation labels retained.
      - unique_snrs: list of unique SNR values retained.
    """
    print(f"Loading RadioML 2016.10A dataset from: {filepath}")
    with open(filepath, 'rb') as f:
        # The data was pickled in Python2, so specify encoding for compatibility
        data_dict = pickle.load(f, encoding='latin1')

    radioml_dict = {}
    # Map raw analog labels to simplified labels
    analog_mod_map = {
        'AM-DSB': 'AM',    # Double Sideband AM
        'AM-SSB': 'AM',    # Single Sideband AM (map to AM)
        'WBFM':   'FM'     # Wideband FM (corrected from B-FM to WBFM)
    }
    all_mod_names = set()

    for (mod, snr), signals in data_dict.items():
        mod_str = mod if isinstance(mod, str) else mod.decode('utf-8', errors='ignore')
        all_mod_names.add(mod_str)

        process_signal = False
        if filter_analog:
            # Only process if it's an analog modulation in our map
            if mod_str in analog_mod_map:
                label = analog_mod_map.get(mod_str)
                process_signal = True
        else:
            # Process all modulations if not filtering
            label = mod_str # Use original modulation name
            process_signal = True

        if process_signal:
             # Convert I/Q data from separate I and Q arrays to complex
            # signals is of shape (num_examples, 2, num_samples)
            complex_signals = []
            for i in range(signals.shape[0]):
                # I channel is signals[i,0,:], Q channel is signals[i,1,:]
                complex_signal = signals[i, 0, :] + 1j * signals[i, 1, :]
                complex_signals.append(complex_signal)
            key = (label, snr)
            radioml_dict.setdefault(key, []).extend(complex_signals)


    unique_mods = sorted({k[0] for k in radioml_dict.keys()})
    unique_snrs = sorted({k[1] for k in radioml_dict.keys()})
    print(f"Found {len(all_mod_names)} raw modulation types.")
    print(f"Loaded dataset contains {len(unique_mods)} modulation classes and {len(unique_snrs)} SNR values.")

    # Simplified one-line summary (no per-mod/per-SNR dump)
    total_samples = sum(len(v) for v in radioml_dict.values())
    print(f"Dataset summary: {len(unique_mods)} mods, {len(unique_snrs)} SNRs, {total_samples:,} total samples.")

    return radioml_dict, unique_mods, unique_snrs
    
# Train/Val/Test Splitting Helper

def radioml_splits(
    data_path,
    seed=42,
    test_size=0.3,
    val_size=0.5,
    flatten=True,
):
    """
    Load RadioML data, convert to X, y, snr arrays, and create
    train/val/test splits.

    Args:
        data_path: path to the .pkl dataset
        seed: random_state for reproducibility
        test_size: fraction of data to hold out for test+val
        val_size: fraction of (temp) data to use as val (e.g., 0.5 -> 50/50 val/test)
        flatten: if True, returns X_* as 2D (samples, features) for classical ML

    Returns:
        A dict with:
            X_train, X_val, X_test
            y_train, y_val, y_test
            snr_train, snr_val, snr_test
            mod_classes, snr_values
    """
    radioml_data, mod_classes, snr_values = load_radioml_pkl_dataset(data_path)

    X = []
    y = []
    snr_list = []

    # radioml_data is keyed by (mod, snr)
    for (mod, snr), signals in radioml_data.items():
        # mod is a string label like '8PSK', 'QAM16', etc.
        mod_idx = mod_classes.index(mod)

        for complex_signal in signals:
            # complex_signal shape: (N,) complex -> split into I/Q
            iq_matrix = np.vstack((complex_signal.real, complex_signal.imag))  # shape (2, N)
            X.append(iq_matrix)
            y.append(mod_idx)
            snr_list.append(snr)

    X = np.array(X)            # (num_samples, 2, N)
    y = np.array(y)
    snr_list = np.array(snr_list)

    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp, snr_train, snr_temp = train_test_split(
        X, y, snr_list,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Second split: val vs test from temp set
    X_val, X_test, y_val, y_test, snr_val, snr_test = train_test_split(
        X_temp, y_temp, snr_temp,
        test_size=val_size,
        random_state=seed,
        stratify=y_temp,
    )

    if flatten:
        # For classical ML: flatten IQ into a single feature vector
        n_train = X_train.shape[0]
        n_val   = X_val.shape[0]
        n_test  = X_test.shape[0]

        X_train_flat = X_train.reshape(n_train, -1)
        X_val_flat   = X_val.reshape(n_val, -1)
        X_test_flat  = X_test.reshape(n_test, -1)

        X_train, X_val, X_test = X_train_flat, X_val_flat, X_test_flat

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "snr_train": snr_train,
        "snr_val": snr_val,
        "snr_test": snr_test,
        "mod_classes": np.array(mod_classes),
        "snr_values": np.array(snr_values),
    }

