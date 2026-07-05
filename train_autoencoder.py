"""
train_autoencoder.py
----------------------
Trains the SpectrogramAutoencoder on clean-only spectrograms, then evaluates
reconstruction error on the mixed test set (clean/jammed/interference) to
show that anomalies produce higher reconstruction error.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autoencoder_model import SpectrogramAutoencoder, pad_to_even_dims

DATASET_PATH = "jamwatch_dataset.npz"
MODEL_SAVE_PATH = "jamwatch_autoencoder.pt"
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def load_data():
    data = np.load(DATASET_PATH)
    train_clean = torch.tensor(data["train_clean"], dtype=torch.float32).unsqueeze(1)  # add channel dim
    test_specs = torch.tensor(data["test_specs"], dtype=torch.float32).unsqueeze(1)
    test_labels = torch.tensor(data["test_labels"], dtype=torch.long)

    train_clean = pad_to_even_dims(train_clean)
    test_specs = pad_to_even_dims(test_specs)

    return train_clean, test_specs, test_labels


def train():
    train_clean, test_specs, test_labels = load_data()
    print(f"Training set: {train_clean.shape}, Test set: {test_specs.shape}")

    train_loader = DataLoader(TensorDataset(train_clean), batch_size=BATCH_SIZE, shuffle=True)

    model = SpectrogramAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print(f"\nTraining for {EPOCHS} epochs on CLEAN signals only...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{EPOCHS} -- reconstruction loss: {epoch_loss:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # ---- Evaluate: per-sample reconstruction error on the mixed test set ----
    model.eval()
    with torch.no_grad():
        reconstructed = model(test_specs)
        per_sample_error = torch.mean((reconstructed - test_specs) ** 2, dim=(1, 2, 3))

    per_sample_error = per_sample_error.numpy()
    labels = test_labels.numpy()
    label_names = {0: "clean", 1: "jammed", 2: "interference"}

    print("\nReconstruction error by class (lower = model is confident it's normal):")
    for label_id, name in label_names.items():
        mask = labels == label_id
        mean_err = per_sample_error[mask].mean()
        std_err = per_sample_error[mask].std()
        print(f"  {name:15s}: mean={mean_err:.5f}  std={std_err:.5f}")

    np.savez(
        "jamwatch_eval_results.npz",
        errors=per_sample_error,
        labels=labels,
    )
    print("\nSaved evaluation results to jamwatch_eval_results.npz")


if __name__ == "__main__":
    train()
