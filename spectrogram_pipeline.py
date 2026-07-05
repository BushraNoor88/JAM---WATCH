"""
spectrogram_pipeline.py
------------------------
Converts raw IQ signal samples into spectrograms using the Short-Time
Fourier Transform (STFT). This is the standard first step in any RF deep
learning pipeline: it turns a 1D complex time-series into a 2D
time-vs-frequency "image" that a CNN or autoencoder can process.

Why STFT?
A single FFT tells you frequency content but loses *when* it happened.
STFT slides a window across the signal and computes an FFT on each window,
giving you a time-frequency picture -- exactly what shows a jammer
appearing as a bright horizontal line, or interference as a noisy band.
"""

import numpy as np
from scipy.signal import stft


def iq_to_spectrogram(
    iq_signal: np.ndarray,
    sample_rate: int,
    nperseg: int = 128,
    noverlap: int = 96,
) -> np.ndarray:
    """
    Convert a complex IQ signal into a log-magnitude spectrogram.

    Parameters
    ----------
    iq_signal : complex-valued 1D array of raw IQ samples
    sample_rate : samples per second
    nperseg : STFT window length (bigger = better frequency resolution,
              worse time resolution)
    noverlap : overlap between windows (more overlap = smoother time axis)

    Returns
    -------
    2D numpy array (frequency_bins x time_bins), log-scaled magnitude,
    normalized to roughly [0, 1] -- ready to feed into a CNN as a
    single-channel "image".
    """
    freqs, times, Zxx = stft(
        iq_signal,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,  # IQ data is complex -> need both pos/neg freqs
    )

    magnitude = np.abs(Zxx)
    log_magnitude = np.log1p(magnitude)  # log1p avoids log(0) issues

    # Normalize to [0, 1] per-sample so the model sees consistent scale
    min_val, max_val = log_magnitude.min(), log_magnitude.max()
    normalized = (log_magnitude - min_val) / (max_val - min_val + 1e-8)

    # Shift zero frequency to the center for a more intuitive plot/image
    normalized = np.fft.fftshift(normalized, axes=0)
    freqs = np.fft.fftshift(freqs)

    return normalized, freqs, times


if __name__ == "__main__":
    from signal_generator import generate_clean_signal, SAMPLE_RATE

    signal = generate_clean_signal(seed=1)
    spec, freqs, times = iq_to_spectrogram(signal, SAMPLE_RATE)
    print(f"Spectrogram shape (freq_bins x time_bins): {spec.shape}")
    print(f"Frequency range: {freqs.min():.0f} Hz to {freqs.max():.0f} Hz")
    print(f"Time range: {times.min()*1000:.1f} ms to {times.max()*1000:.1f} ms")
