"""
Model comparison: rule baseline vs correlation matching vs SVM vs Random Forest.

Trains the two ML models, races all four on the shared seed-42 held-out test
set, prints the F1-macro table + per-class report for the winner, and saves three
figures to reports/:

    reports/confusion_svm.png   — row-normalized confusion matrix (SVM)
    reports/confusion_rf.png    — row-normalized confusion matrix (RF)
    reports/rf_importances.png  — RF feature importance over the wavenumber axis

Run from the project root:
    python scripts/compare_models.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import WAVENUMBER_GRID, compare_models

REPORTS_DIR = Path("reports")

# Functional-group regions to shade on the importance plot, so the forest's
# attention can be read as chemistry rather than anonymous feature indices.
FG_REGIONS = [
    (2800, 3000, "C–H stretch"),
    (1650, 1750, "C=O stretch"),
    (1000, 1300, "C–O / C–N"),
]


def shorten(label: str, width: int = 22) -> str:
    """Trim the long OpenSpecy class strings so axis ticks stay readable."""
    return label if len(label) <= width else label[: width - 1] + "…"


def plot_confusion(y_test, y_pred, labels, title, out_path):
    """Save a row-normalized confusion matrix (each row = per-class error rate)."""
    short = [shorten(l) for l in labels]
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=labels,            # fix row/col order so SVM and RF plots line up
        display_labels=short,
        normalize="true",         # divide each row by its true-class count
        cmap="Blues",
        values_format=".2f",
        xticks_rotation=90,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_rf_importances(rf, out_path):
    """Save the RF's per-feature importance plotted against the wavenumber grid."""
    importances = rf.feature_importances_  # one weight per of the 851 features
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(WAVENUMBER_GRID, importances, lw=0.8, color="darkgreen")

    for lo, hi, name in FG_REGIONS:
        ax.axvspan(lo, hi, color="orange", alpha=0.15)
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.9, name,
                ha="center", va="top", fontsize=8, color="darkorange")

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("RF feature importance")
    ax.set_title("Random Forest feature importance across the IR spectrum")
    ax.invert_xaxis()  # conventional IR presentation: high wavenumber on the left
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    results = compare_models()
    y_test = results["y_test"]
    preds = results["predictions"]

    # Winner = highest F1-macro; show its full per-class breakdown.
    winner = max(results["f1_macro"], key=results["f1_macro"].get)
    print(f"\n=== Per-class report — winner: {winner} ===")
    print(classification_report(y_test, preds[winner], zero_division=0))

    # Stable class order for both confusion matrices.
    labels = sorted(np.unique(y_test))

    print("\nSaving figures to reports/ …")
    plot_confusion(y_test, preds["svm"], labels,
                   "Confusion matrix — SVM (row-normalized)",
                   REPORTS_DIR / "confusion_svm.png")
    plot_confusion(y_test, preds["random_forest"], labels,
                   "Confusion matrix — Random Forest (row-normalized)",
                   REPORTS_DIR / "confusion_rf.png")
    plot_rf_importances(results["models"]["random_forest"],
                        REPORTS_DIR / "rf_importances.png")


if __name__ == "__main__":
    main()
