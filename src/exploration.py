## Functions for Exploring the data
from matplotlib import pyplot as plt
from matplotlib import animation 
import numpy as np
import random
from typing import Dict, List, Tuple, Optional

def _normalize_label(x):
    return x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else str(x)

def visualize_modulations(
    data: Dict[Tuple[str, int], list],
    mods: Optional[List[str]] = None,
    snr: int = 18,
    num_samples: int = 128,
    analog_only: bool = False,
    seed: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Plot I/Q time traces (left) and constellation (right) for each mod at a fixed SNR.
    Returns (fig, axs) so callers can save/modify the figure.
    """
    rng = random.Random(seed)

    # Discover available mods at this SNR
    all_mods = sorted({_normalize_label(m) for (m, s) in data.keys() if s == snr})
    if analog_only:
        all_mods = [m for m in all_mods if m in ("AM", "FM")]

    if mods is None:
        mods = all_mods[:4]

    if not mods:
        print(f"No matching modulations found at SNR={snr}.")
        return None, None

    n_mods = len(mods)
    fig, axs = plt.subplots(n_mods, 2, figsize=(figsize[0], 3 * n_mods))
    if n_mods == 1:
        axs = axs.reshape(1, -1)

    for i, mod in enumerate(mods):
        # find signals for (mod, snr)
        match = [(k, v) for k, v in data.items() if _normalize_label(k[0]) == mod and k[1] == snr]
        if not match or not match[0][1]:
            axs[i, 0].set_visible(False)
            axs[i, 1].set_visible(False)
            print(f"Skipping {mod}@{snr} dB (not found).")
            continue

        signals_list = match[0][1]
        sig = rng.choice(signals_list)  # complex 1D array
        I, Q = np.real(sig), np.imag(sig)

        L = min(num_samples, len(I))

        # time-domain
        axs[i, 0].plot(I[:L], label="I")
        axs[i, 0].plot(Q[:L], label="Q")
        axs[i, 0].set_title(f"{mod} @ {snr} dB — Time Domain")
        axs[i, 0].set_xlabel("Sample")
        axs[i, 0].set_ylabel("Amplitude")
        axs[i, 0].legend()
        axs[i, 0].grid(True, linestyle="--", alpha=0.6)

        # constellation
        axs[i, 1].scatter(I, Q, s=12, alpha=0.7)
        axs[i, 1].set_title(f"{mod} @ {snr} dB — Constellation")
        axs[i, 1].set_xlabel("I")
        axs[i, 1].set_ylabel("Q")
        axs[i, 1].axis("equal")
        axs[i, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig, axs

def animate_bpsk_decision(
    signal: np.ndarray,
    key: Optional[Tuple[str, int]] = None,
    interval: int = 80,
    threshold: float = 0.0,
    max_bits_show: int = 32,
):
    """
    Animate constellation with a vertical decision boundary at I=threshold.
    Returns the matplotlib.animation.FuncAnimation object.
    """
    mod, snr = (key if key is not None else ("BPSK", ""))
    mod = _normalize_label(mod)

    # Accept complex (N,), or (2,N) as [I;Q]
    if signal.ndim == 1 and np.iscomplexobj(signal):
        I, Q = signal.real, signal.imag
    elif signal.ndim == 2 and signal.shape[0] == 2:
        I, Q = signal[0], signal[1]
    else:
        print(f"Unexpected signal shape for {mod}@{snr}.")
        return None
    
    # --- make BPSK clusters pop (single, ordered pass) ---
    
    # 1) Auto phase alignment: rotate so mean symbol lies on +I axis
    z   = (I + 1j*Q)
    phi = np.angle(np.mean(z))
    z   = z * np.exp(-1j * phi)
    I, Q = z.real, z.imag
    if np.mean(I) < 0:  # put dominant cluster at +I
        I = -I; Q = -Q
    
    # 2) AGC (visual): scale to unit RMS, boost a bit
    rms  = (np.mean(I**2 + Q**2) + 1e-12) ** 0.5
    gain = 3.0  # try 2–5
    I = gain * I / rms
    Q = gain * Q / rms
    
    # 3) Symbol decimation (if oversampled)
    sps = 8      # try 4 or 8
    I = I[::sps]
    Q = Q[::sps]
    
    # Decisions after processing
    bits = (I >= 0.0).astype(int)



    # plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    
    ms, al = 32, 0.95
    p = np.percentile(np.c_[I, Q], [1, 99], axis=0)
    padx = 0.1 * (p[1,0] - p[0,0] + 1e-9)
    pady = 0.1 * (p[1,1] - p[0,1] + 1e-9)
    ax.set_xlim(p[0,0]-padx, p[1,0]+padx)
    ax.set_ylim(p[0,1]-pady, p[1,1]+pady)
    
    ax.set_title(f"{mod} @ {snr} dB — Constellation Animation" if snr != "" else f"{mod} — Constellation Animation")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.axvline(threshold, linestyle="--", linewidth=1)
    
    scat0 = ax.scatter([], [], s=ms, alpha=al)
    scat1 = ax.scatter([], [], s=ms, alpha=al)
    txt   = ax.text(
        0.02, 0.02, "",
        transform=ax.transAxes, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
    )

    def init():
        scat0.set_offsets(np.empty((0, 2)))
        scat1.set_offsets(np.empty((0, 2)))
        txt.set_text("")
        return scat0, scat1, txt

    def update(f):
        I_f, Q_f, b_f = I[:f], Q[:f], bits[:f]
        mask0 = (b_f == 0)
        off0 = np.column_stack((I_f[mask0], Q_f[mask0])) if mask0.any() else np.empty((0, 2))
        mask1 = ~mask0
        off1 = np.column_stack((I_f[mask1], Q_f[mask1])) if mask1.any() else np.empty((0, 2))
        scat0.set_offsets(off0)
        scat1.set_offsets(off1)
        show = b_f[-max_bits_show:] if f > 0 else []
        txt.set_text("I>=0 decisions: " + "".join(map(str, show)))
        return scat0, scat1, txt

    ani = animation.FuncAnimation(fig, update, frames=len(I), init_func=init, blit=True, interval=interval)
    plt.close(fig)
    return ani
