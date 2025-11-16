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
def splits(pkl_path, split_dir, seed=42, train_frac=0.70, val_frac=0.15, test_frac=0.15, force_recompute=False):
    """
    Load train/val/test splits if they already exist.
    If they do not exist OR force_recompute=True, regenerate the splits,
    save them to disk, and return the numpy arrays.

    pkl_path:     Path to RadioML2016.10A .pkl dataset
    split_dir:    Directory where npy split files will be stored
    """

    os.makedirs(split_dir, exist_ok=True)

    # Expected files if splits already exist
    files = [
        "X_train.npy", "y_train.npy", "snr_train.npy",
        "X_val.npy",   "y_val.npy",   "snr_val.npy",
        "X_test.npy",  "y_test.npy",  "snr_test.npy",
    ]

    paths = [os.path.join(split_dir, f) for f in files]

    # ----------------------------------------------------------------------
    # 1) If split files exist, load and return them
    # ----------------------------------------------------------------------
    if all(os.path.exists(p) for p in paths) and not force_recompute:
        print("[INFO] Using existing splits from:", split_dir)

        X_train = np.load(paths[0]); y_train = np.load(paths[1]); snr_train = np.load(paths[2])
        X_val   = np.load(paths[3]); y_val   = np.load(paths[4]); snr_val   = np.load(paths[5])
        X_test  = np.load(paths[6]); y_test  = np.load(paths[7]); snr_test  = np.load(paths[8])

        return (
            X_train, y_train, snr_train,
            X_val,   y_val,   snr_val,
            X_test,  y_test,  snr_test
        )

    # ----------------------------------------------------------------------
    # 2) Otherwise load raw dataset and create splits
    # ----------------------------------------------------------------------
    print("[INFO] No splits found. Creating train/val/test now...")

    data_dict, mod_classes, snr_values = load_radioml_pkl_dataset(pkl_path)

    X_list, y_list, snr_list = [], [], []

    # Convert all complex samples into (2, N) IQ matrices
    for (mod, snr), signals in data_dict.items():
        for complex_signal in signals:
            iq = np.vstack((complex_signal.real, complex_signal.imag))
            X_list.append(iq)
            y_list.append(mod)
            snr_list.append(snr)

    X = np.array(X_list)
    y = np.array(y_list)
    snr_list = np.array(snr_list)

    # ----------------------------------------------------------------------
    # 3) Perform 70/15/15 split
    # ----------------------------------------------------------------------
    # First split: Train (70%) and Temp (30%)
    X_train, X_temp, y_train, y_temp, snr_train, snr_temp = train_test_split(
        X, y, snr_list,
        test_size=(1 - train_frac),
        random_state=seed,
        stratify=y
    )

    # Second split: Temp -> Val (15%) + Test (15%)
    relative_test = test_frac / (val_frac + test_frac)  # 0.15 / 0.30 = 0.5

    X_val, X_test, y_val, y_test, snr_val, snr_test = train_test_split(
        X_temp, y_temp, snr_temp,
        test_size=relative_test,
        random_state=seed,
        stratify=y_temp
    )

    # ----------------------------------------------------------------------
    # 4) Save splits
    # ----------------------------------------------------------------------
    print("[INFO] Saving new splits to:", split_dir)

    np.save(os.path.join(split_dir, "X_train.npy"), X_train)
    np.save(os.path.join(split_dir, "y_train.npy"), y_train)
    np.save(os.path.join(split_dir, "snr_train.npy"), snr_train)

    np.save(os.path.join(split_dir, "X_val.npy"), X_val)
    np.save(os.path.join(split_dir, "y_val.npy"), y_val)
    np.save(os.path.join(split_dir, "snr_val.npy"), snr_val)

    np.save(os.path.join(split_dir, "X_test.npy"), X_test)
    np.save(os.path.join(split_dir, "y_test.npy"), y_test)
    np.save(os.path.join(split_dir, "snr_test.npy"), snr_test)

    return (
        X_train, y_train, snr_train,
        X_val,   y_val,   snr_val,
        X_test,  y_test,  snr_test
    )

