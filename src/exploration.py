## Functions for Exporing the date
import matplotlib.pyplot as plt
import numpy as np
import random
def visualize_modulations(data, mods=None, snr=18, num_samples=128): # Reduced num_samples to match RML2016.10A
    """
    Visualize one example of multiple modulation types at a fixed SNR.

    Parameters:
        data : dict
            Dictionary with keys (mod, snr) and values as arrays of signals.
        mods : list
            List of modulation types to show. If None, pick first 4 automatically.
        snr : int
            The SNR level to use for all mods.
        num_samples : int
            Number of time-domain samples to show.
    """
    # Pick modulations automatically if not provided
    if mods is None:
        # Decode keys if they are bytes and filter for analog modulations if needed
        all_mods_raw = sorted(set(m.decode('utf-8', errors='ignore') if isinstance(m, bytes) else m for m, _ in data.keys()))
        # Assuming you only want analog modulations as loaded previously
        analog_mods = [m for m in all_mods_raw if m in ['AM', 'FM']] # Filter for simplified analog mods
        mods = analog_mods[:4] # Take up to the first 4 analog mods if available


    n_mods = len(mods)
    if n_mods == 0:
        print("No analog modulations found in the dataset to visualize.")
        return

    fig, axs = plt.subplots(n_mods, 2, figsize=(12, 3*n_mods))
    # Adjust axs indexing for single row case
    if n_mods == 1:
        axs = axs.reshape(1, -1)


    for i, mod in enumerate(mods):
        # Find the actual keys in data dictionary that match the simplified mod and snr
        # The keys are tuples of (original_mod_bytes or str, snr)
        matching_keys = [k for k in data.keys() if (k[0].decode('utf-8', errors='ignore') if isinstance(k[0], bytes) else k[0]) == mod and k[1] == snr]

        if not matching_keys:
            print(f"Skipping visualization for {mod} at {snr} dB: not found in dataset with this key structure.")
            continue

        # Assume there is at least one matching key and pick the first one
        key = matching_keys[0]

        # Pick a random signal (which is a complex numpy array in radioml_data)
        signals_list = data[key]
        if not signals_list:
            print(f"No signals found for {mod} at {snr} dB.")
            continue
        sig_complex = random.choice(signals_list) # This is a 1D array of complex numbers

        # Extract I and Q components from the complex signal
        I = sig_complex.real
        Q = sig_complex.imag

        # --- Time-domain ---
        axs[i, 0].plot(I[:num_samples], label="I", color="tab:blue")
        axs[i, 0].plot(Q[:num_samples], label="Q", color="tab:orange")
        axs[i, 0].set_title(f"{mod} @ {snr} dB — Time Domain")
        axs[i, 0].set_xlabel("Sample Index")
        axs[i, 0].set_ylabel("Amplitude")
        axs[i, 0].legend(loc="upper right")
        axs[i, 0].grid(True, linestyle="--", alpha=0.6)

        # --- Constellation ---
        axs[i, 1].scatter(I, Q, alpha=0.6, s=15, color="tab:blue", edgecolor="none")
        axs[i, 1].set_title(f"{mod} @ {snr} dB — Constellation")
        axs[i, 1].set_xlabel("I")
        axs[i, 1].set_ylabel("Q")
        axs[i, 1].grid(True, linestyle="--", alpha=0.6)
        axs[i, 1].axis("equal")

    plt.tight_layout()
    plt.show()

def animate_bpsk_decision(signal, key=None, interval=80, threshold=0.0, max_bits_show=32):
    """
    Animated BPSK constellation with decision regions and live bit decoding.
      - Left of I=0 => bit 0  (you can swap mapping if you prefer)
      - Right of I=0 => bit 1
    """
    # unpack & prep
    mod, snr = (key if key is not None else ("BPSK", ""))
    # Decode mod if it's bytes, handle if it's already str or np.str_
    mod = mod.decode('utf-8', errors='ignore') if isinstance(mod, bytes) else str(mod)

    # Ensure signal is a 2D array (channels, samples) if it's a complex 1D array
    if signal.ndim == 1 and np.iscomplexobj(signal):
        I, Q = signal.real, signal.imag
    elif signal.ndim == 2 and signal.shape[0] == 2:
        I, Q = signal[0], signal[1]
    else:
        print(f"Warning: Unexpected signal shape or type for {mod} at {snr} dB. Skipping animation.")
        return None # Return None if signal format is unexpected

    bits = (I >= threshold).astype(int)      # 1 where I>=0, else 0

    # figure/axes
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect("equal")
    pad = 0.5
    # Adjust limits based on actual signal data
    ax.set_xlim(I.min()-pad, I.max()+pad)
    ax.set_ylim(Q.min()-pad, Q.max()+pad)
    ax.set_title(f"{mod} @ {snr} dB — Constellation Animation" if snr!="" else f"{mod} — Constellation Animation")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True, linestyle="--", alpha=0.6)

    # decision boundary + shaded regions (still show for context, though not BPSK)
    ax.axvline(threshold, linestyle="--", linewidth=1)
    ax.axvspan(ax.get_xlim()[0], threshold, alpha=0.08)   # region for bit 0 (I<th)
    ax.axvspan(threshold, ax.get_xlim()[1], alpha=0.08)   # region for bit 1 (I>=th)

    # static labels for regions (adjusting for non-BPSK context)
    yl = ax.get_ylim()
    xmid_left  = (ax.get_xlim()[0] + threshold) / 2
    xmid_right = (threshold + ax.get_xlim()[1]) / 2
    ax.text(xmid_left,  0.9*yl[1], "I < 0", ha="center", va="top")
    ax.text(xmid_right, 0.9*yl[1], "I >= 0", ha="center", va="top")


    # two scatters so we can color by bit (based on I>=threshold)
    scat0 = ax.scatter([], [], s=22, alpha=0.85)  # points where I < threshold
    scat1 = ax.scatter([], [], s=22, alpha=0.85)  # points where I >= threshold


    # live-decoded bit string display (adjusting for non-BPSK context)
    txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, ha="left", va="bottom",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    def init():
        scat0.set_offsets(np.empty((0,2)))
        scat1.set_offsets(np.empty((0,2)))
        txt.set_text("")
        return scat0, scat1, txt

    def update(f):
        # use first f samples
        I_f, Q_f, b_f = I[:f], Q[:f], bits[:f]
        # split by bit
        if f == 0:
            off0 = np.empty((0,2)); off1 = np.empty((0,2))
        else:
            mask0 = (b_f == 0)
            mask1 = ~mask0
            off0 = np.column_stack((I_f[mask0], Q_f[mask0])) if mask0.any() else np.empty((0,2))
            off1 = np.column_stack((I_f[mask1], Q_f[mask1])) if mask1.any() else np.empty((0,2))

        scat0.set_offsets(off0)
        scat1.set_offsets(off1)

        # show last few "decision" values as a string
        show = b_f[-max_bits_show:] if f>0 else []
        txt.set_text("I>=0 decisions: " + "".join(map(str, show)) if len(show) else "I>=0 decisions: ")
        return scat0, scat1, txt

    ani = animation.FuncAnimation(fig, update, frames=len(I), init_func=init,
                                  blit=True, interval=interval)
    plt.close(fig)
    return ani
