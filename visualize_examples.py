"""
visualize_examples.py
----------------------
Generates a few example signals of each type and plots their spectrograms
side by side, so you can *see* what a jammer or interference source looks
like in the time-frequency domain before writing any model code.
"""

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np

from signal_generator import (
    generate_clean_signal,
    generate_jammed_signal,
    generate_interference_signal,
    SAMPLE_RATE,
)
from spectrogram_pipeline import iq_to_spectrogram


def plot_examples(save_path: str = "spectrogram_examples.png") -> None:
    signal_types = {
        "Clean signal": generate_clean_signal,
        "Jammed (CW tone jammer)": generate_jammed_signal,
        "Interference (wideband noise)": generate_interference_signal,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (title, generator_fn) in zip(axes, signal_types.items()):
        signal = generator_fn(seed=42)
        spec, freqs, times = iq_to_spectrogram(signal, SAMPLE_RATE)

        im = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            extent=[times.min() * 1000, times.max() * 1000, freqs.min(), freqs.max()],
            cmap="viridis",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")

    fig.suptitle(
        "RF spectrograms: clean vs. jammed vs. interference",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(im, ax=axes, label="Normalized log-magnitude", shrink=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    plot_examples("spectrogram_examples.png")
