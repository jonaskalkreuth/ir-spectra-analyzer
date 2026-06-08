"""
Validation script for step 16 (B1b-1b-i): confirm the existing preprocessing
chain (crop → baseline → smooth → normalize) behaves sensibly on REAL OpenSpecy
FTIR spectra before we build the ML half on top of it.

This is a throwaway/validation artefact, not a shipped module. It samples a few
spectra across material classes, characterises their raw format, runs the chain,
and plots before/after so we can eyeball that nothing blows up.

Run from the project root:
    python scripts/validate_preprocessing.py
"""

import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.data_acquisition import load_openspecy
from src.preprocessing import baseline_correct, smooth, normalize


# Classes to sample — deliberately spanning very different spectral shapes so the
# chain is stressed on more than one kind of curve:
#   poly(propylene) → sharp, well-separated C-H peaks
#   mineral         → broad Si-O / carbonate humps
#   organic matter  → messy, heterogeneous catch-all
SAMPLE_CLASSES = [
    "poly(propylene)",
    "mineral",
    "organic matter",
]
PER_CLASS = 2


def sample_spectra(
    n_per_class: int = PER_CLASS,
    classes: list[str] = SAMPLE_CLASSES,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """
    Pull a small stratified sample of spectra from the OpenSpecy library.

    Returns a list of (label, wavenumbers, intensities) records — one shared
    wavenumber axis is repeated per record so each is a self-contained spectrum.
    """
    wavenumbers, spectra, labels = load_openspecy()

    records: list[tuple[str, np.ndarray, np.ndarray]] = []
    for cls in classes:
        # labels is a pandas Series; this boolean mask gives the positional
        # indices of every row whose material_class matches `cls`.
        idx = np.flatnonzero((labels == cls).to_numpy())
        if idx.size == 0:
            print(f"[sample] WARNING: no spectra found for class {cls!r}, skipping.")
            continue

        chosen = idx[:n_per_class]
        for row in chosen:
            # spectra[row] is one spectrum's intensities over the shared axis.
            records.append((cls, wavenumbers, spectra[row]))

    print(f"[sample] Selected {len(records)} spectra across {len(classes)} classes.")
    return records


CROP_LO = 600
CROP_HI = 4000


def characterize(records: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    """
    Print the raw format of each sampled spectrum: NaN fraction, the measured
    wavenumber span, and the intensity range. This is the 'know your data before
    you touch it' step — it tells us how big the NaN-padding problem is and
    whether the intensities are already normalized.
    """
    print("\n=== Raw format ===")
    for label, wn, intensity in records:
        nan_mask = np.isnan(intensity)
        nan_frac = nan_mask.mean()  # mean of a boolean array = fraction that are True

        finite = ~nan_mask
        # nanmin/nanmax ignore NaN; plain min/max would return NaN if any is present.
        lo_wn, hi_wn = wn[finite].min(), wn[finite].max()
        lo_i, hi_i = np.nanmin(intensity), np.nanmax(intensity)

        print(
            f"  {label:<20} NaN {nan_frac:5.1%} | "
            f"measured {lo_wn:6.0f}–{hi_wn:6.0f} cm⁻¹ | "
            f"intensity {lo_i:.3f}–{hi_i:.3f}"
        )


def crop(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    lo: float = CROP_LO,
    hi: float = CROP_HI,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Restrict a spectrum to the [lo, hi] cm⁻¹ window. A boolean mask works
    regardless of whether the axis is ascending or descending.
    """
    keep = (wavenumbers >= lo) & (wavenumbers <= hi)
    return wavenumbers[keep], intensities[keep]


def preprocess_one(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    lo: float = CROP_LO,
    hi: float = CROP_HI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the validated chain on a single spectrum:
        finite + window mask  →  baseline_correct  →  smooth  →  normalize

    Returns (wavenumbers, raw_masked, processed) — all on the same cropped axis,
    so 16C can overlay the raw and processed curves point-for-point.
    """
    # One combined mask: inside the window AND actually measured (not NaN).
    # This is the agreed straggler rule — baseline/smooth never see a NaN.
    keep = (wavenumbers >= lo) & (wavenumbers <= hi) & ~np.isnan(intensities)
    wn = wavenumbers[keep]
    raw = intensities[keep]

    corrected = baseline_correct(raw)
    smoothed = smooth(corrected)
    processed = normalize(smoothed)  # normalize LAST: after baseline+smooth change the range

    return wn, raw, processed


def plot_before_after(
    records: list[tuple[str, np.ndarray, np.ndarray]],
    out_path: str = "scripts/validate_preprocessing.png",
) -> None:
    """
    Overlay raw (masked) vs processed for each sampled spectrum, one panel each.
    Saves to a PNG rather than showing interactively, since this runs headless.
    """
    n = len(records)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.4 * n), sharex=True)

    for ax, (label, wn0, intensity0) in zip(axes, records):
        wn, raw, processed = preprocess_one(wn0, intensity0)
        ax.plot(wn, raw, lw=0.8, alpha=0.5, label="raw (masked)")
        ax.plot(wn, processed, lw=1.0, label="processed")
        ax.set_ylabel("intensity")
        ax.set_title(label, fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper right")

    # FTIR convention: wavenumber decreases left→right. Set once; sharex applies it to all.
    axes[0].set_xlim(CROP_HI, CROP_LO)
    axes[-1].set_xlabel("wavenumber (cm⁻¹)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\n[plot] Saved before/after comparison → {out_path}")


if __name__ == "__main__":
    sample = sample_spectra()
    characterize(sample)

    print(f"\n=== After cropping to {CROP_LO}–{CROP_HI} cm⁻¹ ===")
    cropped = [(label, *crop(wn, intensity)) for label, wn, intensity in sample]
    characterize(cropped)

    print("\n=== Running the chain (no NaN should survive) ===")
    for label, wn, intensity in sample:
        w, raw, processed = preprocess_one(wn, intensity)
        n_bad = np.isnan(processed).sum()
        print(
            f"  {label:<20} {w.size} pts | "
            f"processed range {processed.min():.3f}–{processed.max():.3f} | "
            f"NaN in output: {n_bad}"
        )

    plot_before_after(sample)
