"""
Step 22 — Persist the winning model (SVM) with joblib.

Re-fits the tuned SVM on the full training set with `probability=True` (so the
saved model can emit a confidence later), scores it once on the shared seed-42
held-out test set, and dumps a self-describing bundle to:

    models/svm_classifier.joblib

Then reloads it and confirms predictions round-trip, so we never ship a file we
haven't verified loads. Run from the project root:

    python scripts/persist_model.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import load_model, load_split, save_model, train_svm


def main():
    # Re-fit the tuned SVM with probability calibration on (step-22 flag).
    model = train_svm(probability=True)

    # Recompute the held-out F1 on the SAME seed-42 split, to store in the bundle.
    # train_svm already split internally; load_split is deterministic so these are
    # the identical test rows it evaluated on.
    _, X_test, _, y_test = load_split()
    test_f1 = f1_score(y_test, model.predict(X_test), average="macro")

    path = save_model(model, test_f1)
    print(f"\n[persist] saved bundle → {path}  (test F1-macro {test_f1:.3f})")

    # Round-trip check: reload and confirm the loaded model predicts identically,
    # and that probability=True actually took (predict_proba must not raise).
    bundle = load_model(path)
    reloaded = bundle["model"]
    same = np.array_equal(reloaded.predict(X_test), model.predict(X_test))
    proba = reloaded.predict_proba(X_test[:1])  # raises if probability wasn't fit
    print(f"[persist] reload predictions match in-memory model: {same}")
    print(f"[persist] predict_proba works — top confidence on one spectrum: "
          f"{proba.max():.2f} ({bundle['classes'][proba.argmax()]})")


if __name__ == "__main__":
    main()
