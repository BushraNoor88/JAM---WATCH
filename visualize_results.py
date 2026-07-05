"""
visualize_results.py
----------------------
Plots the reconstruction error distributions for clean/jammed/interference
test samples, showing how well the autoencoder separates normal spectrum
from anomalies. This is the key result chart for the README/portfolio.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "jamwatch_eval_results.npz"


def plot_error_distribution(save_path: str = "anomaly_detection_results.png"):
    data = np.load(RESULTS_PATH)
    errors = data["errors"]
    labels = data["labels"]

    label_names = {0: "Clean", 1: "Jammed", 2: "Interference"}
    colors = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram of reconstruction error per class
    for label_id, name in label_names.items():
        mask = labels == label_id
        ax1.hist(errors[mask], bins=20, alpha=0.6, label=name, color=colors[label_id])
    ax1.set_xlabel("Reconstruction error (MSE)")
    ax1.set_ylabel("Count")
    ax1.set_title("Reconstruction error distribution by signal type")
    ax1.legend()

    # Right: box plot for a cleaner "at a glance" comparison
    box_data = [errors[labels == label_id] for label_id in label_names.keys()]
    bp = ax2.boxplot(box_data, labels=list(label_names.values()), patch_artist=True)
    for patch, label_id in zip(bp["boxes"], label_names.keys()):
        patch.set_facecolor(colors[label_id])
        patch.set_alpha(0.6)
    ax2.set_ylabel("Reconstruction error (MSE)")
    ax2.set_title("Error spread by signal type")

    fig.suptitle(
        "JAM//WATCH -- autoencoder anomaly scores (trained on clean signals only)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    plot_error_distribution("anomaly_detection_results.png")
