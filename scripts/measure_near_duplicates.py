"""
Measure how many test spectra have a near-twin in the training set.

An exact-duplicate count can't see near-duplicates: two replicate measurements
of the same material differ by noise but correlate at r ≈ 0.999. A test spectrum
with such a twin in training is close to a lookup, not a generalisation test.

For every row of the seed-42 test set, this computes its maximum Pearson r to
all training rows (processed 851-feature vectors), reports the near-twin share
at three thresholds, then re-scores the saved SVM and the correlation baseline
on the test rows with NO training neighbour above 0.99 and above 0.95.

Run from the project root (needs the cached features and the saved model):

    python scripts/measure_near_duplicates.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import load_model, load_split, predict_correlation_baseline

THRESHOLDS = (0.999, 0.99, 0.95)
NO_TWIN_THRESHOLDS = (0.99, 0.95)
CHUNK = 1000  # test rows per matrix product; keeps the (chunk, n_train) block small


def max_train_correlation(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    For each test row, the highest Pearson r to any training row.

    Row-standardise both matrices (mean 0, unit L2 norm), so a dot product is the
    Pearson r. Done in chunks of test rows to bound memory.
    """
    def _normed(M: np.ndarray) -> np.ndarray:
        Mc = M - M.mean(axis=1, keepdims=True)
        return Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-12)

    train_n, test_n = _normed(X_train), _normed(X_test)
    return np.concatenate([
        (test_n[i:i + CHUNK] @ train_n.T).max(axis=1)
        for i in range(0, len(test_n), CHUNK)
    ])


def main():
    X_train, X_test, y_train, y_test = load_split()
    labels = sorted(np.unique(y_train))  # the 8 real classes, for f1_score

    max_r = max_train_correlation(X_train, X_test)
    print(f"\nTest rows: {len(X_test)}   (train rows: {len(X_train)})\n")
    print("Share of test rows with a training neighbour at:")
    for t in THRESHOLDS:
        print(f"  r ≥ {t:<5}  {np.mean(max_r >= t):6.1%}")

    svm_pred = load_model()["model"].predict(X_test)
    corr_pred = predict_correlation_baseline(X_train, y_train, X_test)

    print("\nF1-macro on test rows with NO training neighbour at:")
    print(f"  {'threshold':<10} {'n':>6} {'SVM':>7} {'corr':>7}")
    for t in NO_TWIN_THRESHOLDS:
        mask = max_r < t
        svm_f1 = f1_score(y_test[mask], svm_pred[mask], labels=labels, average="macro")
        corr_f1 = f1_score(y_test[mask], corr_pred[mask], labels=labels, average="macro")
        print(f"  r ≥ {t:<6} {mask.sum():>6} {svm_f1:>7.3f} {corr_f1:>7.3f}")

    # Small minority classes make the no-twin scores noisy; show the counts.
    for t in NO_TWIN_THRESHOLDS:
        mask = max_r < t
        classes, counts = np.unique(y_test[mask], return_counts=True)
        print(f"\nPer-class n, no neighbour at r ≥ {t}:")
        for c, n in zip(classes, counts):
            print(f"  {n:>5}  {c}")


if __name__ == "__main__":
    main()
