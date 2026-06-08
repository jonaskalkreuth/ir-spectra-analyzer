# Polymer classifier: SVM + Random Forest trained on OpenSpecy FTIR spectra.
# Feature vector: spectrum resampled to a fixed 600–4000 cm⁻¹ grid at 4 cm⁻¹
# steps (851 features), then baseline-corrected, smoothed, and normalized —
# the processed grid IS the feature vector fed to SVM/RF.
# TODO: train/evaluate/persist implemented in steps B1b-1c-ii through B1b-1c-v.

from pathlib import Path

import numpy as np
import pandas as pd

from .data_acquisition import load_openspecy
from .preprocessing import baseline_correct, smooth, normalize


# Feature grid: the step-16 ROI (600–4000 cm⁻¹), NOT the original 400–4000.
# Below ~600 the OpenSpecy FTIR grid is empty NaN-pad; above 4000 is NIR overtone
# territory, not mid-IR fingerprint. 4 cm⁻¹ spacing → 851 points.
GRID_START = 600
GRID_END = 4000
GRID_STEP = 4
WAVENUMBER_GRID = np.arange(GRID_START, GRID_END + GRID_STEP, GRID_STEP)  # 851 points


def resample_to_grid(wavenumbers: np.ndarray, intensities: np.ndarray) -> np.ndarray:
    """
    Interpolate a spectrum onto the fixed wavenumber grid.

    Drops NaN-padded points first: the raw OpenSpecy spectra are padded with NaN
    outside their measured range, and np.interp would otherwise carry those NaNs
    into the output. Grid points beyond a spectrum's measured span are edge-clamped
    by np.interp (flat extension) — acceptable since post-crop coverage is ~100%.

    Returns a 1-D array of length len(WAVENUMBER_GRID) (851).
    """
    finite = ~np.isnan(intensities)
    wn, y = wavenumbers[finite], intensities[finite]

    # np.interp REQUIRES its xp (sample points) to be monotonically increasing;
    # the OpenSpecy axis may be descending, so sort ascending and carry y along.
    order = np.argsort(wn)
    return np.interp(WAVENUMBER_GRID, wn[order], y[order])


def spectrum_to_features(
    wavenumbers: np.ndarray, intensities: np.ndarray
) -> np.ndarray:
    """
    Turn one raw spectrum into its fixed-length feature vector.

    Resample-FIRST (fork 2→B): land the raw spectrum on the uniform 851-point grid,
    THEN run the validated chain on that grid. Because every spectrum now shares the
    exact same axis, savgol's window is a constant physical cm⁻¹ width across the whole
    dataset, and feature column k maps to a fixed wavenumber for every sample.

    Returns the 851-length processed vector that is fed directly to SVM/RF.
    """
    resampled = resample_to_grid(wavenumbers, intensities)
    corrected = baseline_correct(resampled)   # ALS baseline subtract (confirmatory on flat data)
    smoothed = smooth(corrected)              # savgol: constant window now the grid is uniform
    return normalize(smoothed)                # min-max to [0,1] LAST, after the range has shifted


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


# --- Step 18: feature engineering --------------------------------------------


def build_feature_matrix(
    wavenumbers: np.ndarray, spectra: np.ndarray
) -> np.ndarray:
    """
    Resample + preprocess every raw spectrum into the fixed-length feature matrix.

    `wavenumbers` is the single shared axis (same for every row of `spectra`), so
    each row is mapped through `spectrum_to_features` onto the same 851-point grid.

    Returns X of shape (n_spectra, 851), dtype float32.
    """
    # Plain loop + stack: the per-spectrum NaN/sort masking doesn't vectorize
    # cleanly, and 20k iterations is trivial next to the ALS solve inside each.
    rows = [spectrum_to_features(wavenumbers, row) for row in spectra]
    X = np.stack(rows).astype(np.float32)  # float32: half the memory, ample precision for [0,1]

    # Sanity gate — cheap insurance that the matrix is model-ready before we
    # hand it to sklearn (a stray NaN would only surface as a cryptic fit error).
    assert X.shape == (len(spectra), len(WAVENUMBER_GRID)), f"unexpected shape {X.shape}"
    assert not np.isnan(X).any(), "feature matrix contains NaN"
    assert X.min() >= 0.0 and X.max() <= 1.0, f"out of [0,1]: {X.min()}–{X.max()}"
    return X


def save_feature_matrix(
    path: Path, X: np.ndarray, labels: np.ndarray, grid: np.ndarray
) -> None:
    """Cache the feature matrix + labels + grid as one compressed .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=str)  # same object→unicode cast as step 17
    np.savez_compressed(path, X=X, labels=labels, grid=grid)


def build_features(
    train_path: Path = Path("data/openspecy/training_set.npz"),
    out_path: Path = Path("data/openspecy/features.npz"),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the full step-18 pipeline: load step-17 set → build X → cache → return (X, y).

    Hybrid output (fork 4): returns (X, y) in-memory for callers AND caches to
    features.npz so steps 19–21 can np.load without re-running the chain per CV fit.
    """
    data = np.load(train_path)
    wavenumbers, spectra, labels = data["wavenumbers"], data["spectra"], data["labels"]

    print(f"[features] building {len(spectra):,} spectra → {len(WAVENUMBER_GRID)}-d grid …")
    X = build_feature_matrix(wavenumbers, spectra)

    save_feature_matrix(out_path, X, labels, WAVENUMBER_GRID)
    print(f"[features] X={X.shape} {X.dtype}, range {X.min():.3f}–{X.max():.3f}")
    print(f"[features] saved → {out_path}")
    return X, labels
