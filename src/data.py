## File is called in the Notebooks, this handles loading the dataset and preprocessing
import pickle
import numpy as np
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

