# Polymer classifier: SVM + Random Forest trained on OpenSpecy FTIR spectra.
# Feature vector: spectrum resampled to fixed 400–4000 cm⁻¹ grid at 4 cm⁻¹
# steps (~900 features), normalized intensities used directly.
# TODO: implemented in steps B1b-1c-i through B1b-1c-v.

from pathlib import Path

import numpy as np
import pandas as pd

from .data_acquisition import load_openspecy


GRID_START = 400
GRID_END = 4000
GRID_STEP = 4
WAVENUMBER_GRID = np.arange(GRID_START, GRID_END + GRID_STEP, GRID_STEP)  # ~900 points


def resample_to_grid(wavenumbers: np.ndarray, intensities: np.ndarray) -> np.ndarray:
    """
    Interpolate a spectrum onto the fixed wavenumber grid.

    Returns a 1-D array of length len(WAVENUMBER_GRID).
    """
    return np.interp(WAVENUMBER_GRID, np.sort(wavenumbers), intensities[np.argsort(wavenumbers)])


# --- Step 17: training-set assembly ------------------------------------------

MIN_CLASS_COUNT = 300  # classes below this are dropped; natural cliff (9th class = 126)


def count_exact_duplicates(spectra: np.ndarray) -> int:
    """
    Count spectra that exactly repeat an earlier intensity row.

    Backs the "low-replication reference library" claim with a real number,
    measured on the FULL library before any filtering.
    """
    # pandas hashes each row to find repeats and — unlike numpy — treats the
    # NaN-pad columns as equal, so two identically-padded spectra register as
    # duplicates. keep="first" flags every row that has appeared before.
    return int(pd.DataFrame(spectra).duplicated(keep="first").sum())


def drop_exact_duplicates(
    spectra: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop exact-duplicate intensity rows, keeping the first occurrence of each."""
    keep = ~pd.DataFrame(spectra).duplicated(keep="first").to_numpy()
    # boolean-mask both in lockstep so spectra rows stay aligned with their labels
    return spectra[keep], labels[keep]


def filter_classes(
    spectra: np.ndarray, labels: np.ndarray, min_count: int = MIN_CLASS_COUNT
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only classes with at least `min_count` spectra; drop the rare tail.

    Imbalance among the SURVIVING classes is left intact (handled later by
    `class_weight='balanced'`) — this only cuts classes too sparse to learn.
    """
    counts = pd.Series(labels).value_counts()
    keep_classes = counts[counts >= min_count].index
    keep = np.isin(labels, keep_classes)
    return spectra[keep], labels[keep]


def save_training_set(
    path: Path,
    spectra: np.ndarray,
    labels: np.ndarray,
    wavenumbers: np.ndarray,
) -> None:
    """Persist the clean training set as a single compressed .npz archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Cast labels object→fixed-width unicode ('<U...'): object arrays can only be
    # saved by pickling, and np.load refuses pickles by default (allow_pickle=False),
    # so an object-dtype labels array would save but fail to reload.
    labels = np.asarray(labels, dtype=str)
    # one binary file, three named arrays; np.load(path)["spectra"] reads it back
    np.savez_compressed(path, spectra=spectra, labels=labels, wavenumbers=wavenumbers)


def assemble_training_set(
    data_dir: Path = Path("data/openspecy"),
    out_path: Path = Path("data/openspecy/training_set.npz"),
    min_count: int = MIN_CLASS_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full step-17 pipeline: load → measure dupes → drop → class-filter → save.

    Returns the cleaned (wavenumbers, spectra, labels).
    """
    wavenumbers, spectra, labels = load_openspecy(data_dir)
    labels = labels.to_numpy()  # work in numpy so masks stay alignment-safe

    n_start = len(spectra)
    n_dup = count_exact_duplicates(spectra)
    print(
        f"[assemble] duplicate prevalence: {n_dup:,} / {n_start:,} "
        f"({n_dup / n_start:.1%}) exact-duplicate spectra"
    )

    spectra, labels = drop_exact_duplicates(spectra, labels)
    spectra, labels = filter_classes(spectra, labels, min_count)

    print(
        f"[assemble] kept {len(spectra):,} spectra across "
        f"{len(np.unique(labels))} classes (min_count={min_count}):"
    )
    for cls, n in pd.Series(labels).value_counts().items():
        print(f"    {n:>6,}  {cls}")

    save_training_set(out_path, spectra, labels, wavenumbers)
    print(f"[assemble] saved → {out_path}")
    return wavenumbers, spectra, labels
