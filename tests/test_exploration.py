import pytest

np = pytest.importorskip("numpy")

from src.exploration import animate_bpsk_decision


def test_animate_bpsk_decision_rejects_non_positive_gain():
    signal = np.ones(32, dtype=np.complex128)
    with pytest.raises(ValueError):
        animate_bpsk_decision(signal, gain=0.0)

    with pytest.raises(ValueError):
        animate_bpsk_decision(signal, gain=-1)


def test_animate_bpsk_decision_accepts_custom_gain():
    signal = np.exp(1j * np.linspace(0, np.pi, 64))

    # Should not raise when gain is positive and non-default
    ani = animate_bpsk_decision(signal, gain=4.5, sps=1)
    assert ani is not None
