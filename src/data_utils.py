import numpy as np
from typing import Dict, Tuple, List

def dict_to_iq_arrays(
    radioml_data: Dict[Tuple[str, int], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a filtered RadioML dictionary into arrays:
      X         : I/Q samples, shape (N, 2, T)
      y         : modulation labels (as strings), shape (N,)
      snr_list  : SNR values, shape (N,)

    radioml_data is expected to be a dict with keys (mod, snr)
    and values of shape (num_examples, 2, T) or (num_examples, T) complex.
    """
    X = []
    y = []
    snr_list = []

    for (mod, snr), signals in radioml_data.items():
        # signals: iterable of complex vectors or already (2, T) arrays
        for complex_signal in signals:
            # If your signals are complex 1D arrays, uncomment this:
            # iq_matrix = np.vstack((complex_signal.real, complex_signal.imag))
            # X.append(iq_matrix)

            # If they are already shaped (2, T), just append directly:
            X.append(complex_signal)

            y.append(mod)   # mod is already a string label
            snr_list.append(snr)

    X = np.array(X)
    y = np.array(y)
    snr_list = np.array(snr_list)

    return X, y, snr_list
