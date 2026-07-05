"""
dataset_builder.py
--------------------
Generates a bulk dataset of spectrograms for training and evaluation.

Key idea for anomaly detection: the TRAINING set contains ONLY clean signals
(the autoencoder must never see jammed/interference examples during training,
otherwise it would learn to reconstruct them too, defeating the purpose).

The TEST set contains a mix of all three types, so we can measure how well
the trained autoencoder separates clean from anomalous.
"""

import numpy as np

from signal_generator import (
    generate_clean_signal,
    generate_jammed_signal,
    generate_interference_signal,
    SAMPLE_RATE,
)
from spectrogram_pipeline import iq_to_spectrogram


def build_spectrogram_batch(generator_fn, n_samples: int, seed_offset: int = 0) -> np.ndarray:
    """Generate n_samples spectrograms using the given signal generator function."""
    specs = []
    for i in range(n_samples):
        signal = generator_fn(seed=seed_offset + i)
        spec, _, _ = iq_to_spectrogram(signal, SAMPLE_RATE)
        specs.append(spec)
    return np.stack(specs)  # shape: (n_samples, freq_bins, time_bins)


def build_datasets(
    n_train_clean: int = 800,
    n_test_each: int = 100,
):
    """
    Build train/test datasets.

    Returns
    -------
    train_clean : (n_train_clean, F, T) -- clean spectrograms ONLY, for training
    test_specs  : (3 * n_test_each, F, T) -- mixed clean/jammed/interference
    test_labels : (3 * n_test_each,) -- 0=clean, 1=jammed, 2=interference
    """
    print(f"Generating {n_train_clean} clean training spectrograms...")
    train_clean = build_spectrogram_batch(generate_clean_signal, n_train_clean, seed_offset=0)

    print(f"Generating {n_test_each} test spectrograms per class (clean/jammed/interference)...")
    test_clean = build_spectrogram_batch(generate_clean_signal, n_test_each, seed_offset=100_000)
    test_jammed = build_spectrogram_batch(generate_jammed_signal, n_test_each, seed_offset=200_000)
    test_interference = build_spectrogram_batch(
        generate_interference_signal, n_test_each, seed_offset=300_000
    )

    test_specs = np.concatenate([test_clean, test_jammed, test_interference], axis=0)
    test_labels = np.concatenate(
        [
            np.zeros(n_test_each),      # 0 = clean
            np.ones(n_test_each),       # 1 = jammed
            np.full(n_test_each, 2),    # 2 = interference
        ]
    )

    return train_clean, test_specs, test_labels


if __name__ == "__main__":
    train_clean, test_specs, test_labels = build_datasets(n_train_clean=800, n_test_each=100)
    print(f"\ntrain_clean shape: {train_clean.shape}")
    print(f"test_specs shape:  {test_specs.shape}")
    print(f"test_labels shape: {test_labels.shape}")
    print(f"Label counts -> clean: {(test_labels==0).sum()}, "
          f"jammed: {(test_labels==1).sum()}, "
          f"interference: {(test_labels==2).sum()}")

    np.savez(
        "jamwatch_dataset.npz",
        train_clean=train_clean,
        test_specs=test_specs,
        test_labels=test_labels,
    )
    print("\nSaved dataset to jamwatch_dataset.npz")
