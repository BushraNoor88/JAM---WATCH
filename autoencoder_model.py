"""
autoencoder_model.py
----------------------
A small convolutional autoencoder for RF spectrogram anomaly detection.

Trained ONLY on clean spectrograms. Learns to compress a spectrogram down
to a small "bottleneck" representation and reconstruct it back. When shown
a jammed or interference spectrogram (patterns it has never seen), the
reconstruction will be noticeably worse -- that reconstruction error is
our anomaly score.
"""

import torch
import torch.nn as nn


class SpectrogramAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: progressively compress the spectrogram
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # -> (16, 64, 17)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 32, 9)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 16, 5)
            nn.ReLU(),
        )

        # Decoder: mirror of the encoder, expands back to original size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # spectrograms are normalized to [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def pad_to_even_dims(spec: torch.Tensor) -> torch.Tensor:
    """
    Our spectrograms are (128, 33). 33 is odd, which causes rounding
    mismatches through strided conv/deconv layers. We pad the time axis
    to 40 (a clean multiple of 8, since we have 3 stride-2 layers).
    """
    freq_bins, time_bins = spec.shape[-2], spec.shape[-1]
    target_time = 40
    pad_amount = target_time - time_bins
    if pad_amount > 0:
        spec = torch.nn.functional.pad(spec, (0, pad_amount))
    return spec


if __name__ == "__main__":
    model = SpectrogramAutoencoder()
    dummy_input = torch.randn(4, 1, 128, 40)  # batch of 4, 1 channel, padded shape
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")
