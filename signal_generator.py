"""
signal_generator.py
--------------------
Generates synthetic RF signals as complex IQ samples (in-phase + quadrature),
which is the standard raw representation of radio signals.

We simulate:
  1. A "clean" signal: a QPSK-modulated carrier (like a normal digital transmission)
  2. A "jammed" signal: the same carrier with a jamming tone injected
  3. An "interference" signal: the same carrier with wideband noise interference

This gives us three classes to build intuition with before scaling up to a
real dataset like RadioML.
"""

import numpy as np


SAMPLE_RATE = 20_000       # samples per second (simulated)
DURATION_SEC = 0.05        # 50ms snippet per sample
N_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
CARRIER_FREQ = 2_000       # Hz, where our "signal of interest" sits in the band


def _generate_qpsk_baseband(n_symbols: int, samples_per_symbol: int) -> np.ndarray:
    """Generate a QPSK-modulated baseband signal (complex IQ)."""
    # Random symbols: each is one of 4 QPSK constellation points
    bits = np.random.randint(0, 4, n_symbols)
    constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    symbols = constellation[bits]
    # Upsample (repeat each symbol to fill its duration -> simple pulse shaping)
    baseband = np.repeat(symbols, samples_per_symbol)
    return baseband[: n_symbols * samples_per_symbol]


def generate_clean_signal(seed: int | None = None) -> np.ndarray:
    """A clean QPSK signal modulated onto a carrier, plus light thermal noise."""
    if seed is not None:
        np.random.seed(seed)

    samples_per_symbol = 40
    n_symbols = N_SAMPLES // samples_per_symbol + 1
    baseband = _generate_qpsk_baseband(n_symbols, samples_per_symbol)[:N_SAMPLES]

    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    carrier = np.exp(1j * 2 * np.pi * CARRIER_FREQ * t)
    signal = baseband * carrier

    noise = (np.random.randn(N_SAMPLES) + 1j * np.random.randn(N_SAMPLES)) * 0.05
    return signal + noise


def generate_jammed_signal(seed: int | None = None) -> np.ndarray:
    """Clean signal + a strong continuous-wave jamming tone at a nearby frequency."""
    clean = generate_clean_signal(seed)

    jammer_freq = CARRIER_FREQ + 3_500  # jammer sits near, not on top of, the carrier
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    jammer_amplitude = 3.0  # much stronger than the signal -> classic jamming behavior
    jammer = jammer_amplitude * np.exp(1j * 2 * np.pi * jammer_freq * t)

    return clean + jammer


def generate_interference_signal(seed: int | None = None) -> np.ndarray:
    """Clean signal + wideband noise interference (e.g. a broadband noise source)."""
    clean = generate_clean_signal(seed)

    wideband_noise = (
        np.random.randn(N_SAMPLES) + 1j * np.random.randn(N_SAMPLES)
    ) * 1.2
    return clean + wideband_noise


if __name__ == "__main__":
    clean = generate_clean_signal(seed=1)
    jammed = generate_jammed_signal(seed=1)
    interference = generate_interference_signal(seed=1)

    print(f"Generated {N_SAMPLES} IQ samples per snippet at {SAMPLE_RATE} Hz")
    print(f"clean signal power:        {np.mean(np.abs(clean) ** 2):.3f}")
    print(f"jammed signal power:       {np.mean(np.abs(jammed) ** 2):.3f}")
    print(f"interference signal power: {np.mean(np.abs(interference) ** 2):.3f}")
