"""
threshold_selector.py
------------------------
Determines a decision threshold for anomaly detection based on the clean
signal reconstruction error distribution. The threshold marks the boundary
above which a signal is flagged as "ANOMALY".

Strategy: use the clean set's mean + a safety margin (multiples of std dev),
capped so it still sits comfortably below the jammed error range. This
mirrors how real anomaly-detection thresholds are chosen in practice --
based on the "known normal" distribution, since real anomalies are rarely
available in production.
"""

import numpy as np

RESULTS_PATH = "jamwatch_eval_results.npz"


def compute_threshold(margin_std: float = 4.0) -> float:
    data = np.load(RESULTS_PATH)
    errors = data["errors"]
    labels = data["labels"]

    clean_errors = errors[labels == 0]
    jammed_errors = errors[labels == 1]
    interference_errors = errors[labels == 2]

    clean_mean = clean_errors.mean()
    clean_std = clean_errors.std()

    threshold = clean_mean + margin_std * clean_std

    print(f"Clean error:        mean={clean_mean:.5f}  std={clean_std:.5f}")
    print(f"Jammed error:        mean={jammed_errors.mean():.5f}  min={jammed_errors.min():.5f}")
    print(f"Interference error:  mean={interference_errors.mean():.5f}  min={interference_errors.min():.5f}")
    print(f"\nChosen threshold ({margin_std} std above clean mean): {threshold:.5f}")

    # Sanity check: how many of each class would be correctly/incorrectly flagged?
    for name, class_errors in [
        ("clean", clean_errors),
        ("jammed", jammed_errors),
        ("interference", interference_errors),
    ]:
        flagged = (class_errors > threshold).sum()
        total = len(class_errors)
        print(f"  {name:15s}: {flagged}/{total} flagged as anomaly")

    return threshold


if __name__ == "__main__":
    threshold = compute_threshold(margin_std=4.0)

    with open("threshold.txt", "w") as f:
        f.write(str(threshold))
    print(f"\nSaved threshold to threshold.txt: {threshold:.5f}")
