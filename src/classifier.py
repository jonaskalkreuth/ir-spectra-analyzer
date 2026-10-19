# Polymer classifier: SVM + Random Forest trained on OpenSpecy FTIR spectra.
# Feature vector: spectrum resampled to a fixed 600–4000 cm⁻¹ grid at 4 cm⁻¹
# steps (851 features), then baseline-corrected, smoothed, and normalized —
# the processed grid IS the feature vector fed to SVM/RF.

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import (
    RANDOM_STATE,
    RF_PARAM_GRID,
    SVM_PARAM_GRID,
    TEST_SIZE,
    TUNE_SUBSAMPLE,
)
from .data_acquisition import load_openspecy
from .peak_detection import detect_peaks
from .polymer_rules import classify_by_rules
from .preprocessing import baseline_correct, smooth, normalize, to_openspecy_derivative


# Feature grid: the preprocessing ROI (600–4000 cm⁻¹), NOT the original 400–4000.
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
    by np.interp (flat extension). Known limitation: 19.5% of spectra stop below
    3,500 cm⁻¹, so their high-wavenumber end is a flat copy of the last measured value.

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

    Resample-FIRST: land the raw spectrum on the uniform 851-point grid,
    THEN run the validated chain on that grid. Because every spectrum now shares the
    exact same axis, savgol's window is a constant physical cm⁻¹ width across the whole
    dataset, and feature column k maps to a fixed wavenumber for every sample.

    Returns the 851-length processed vector that is fed directly to SVM/RF.
    """
    resampled = resample_to_grid(wavenumbers, intensities)
    corrected = baseline_correct(resampled)   # ALS baseline subtract (confirmatory on flat data)
    smoothed = smooth(corrected)              # savgol: constant window now the grid is uniform
    return normalize(smoothed)                # min-max to [0,1] LAST, after the range has shifted


# --- Training-set assembly ---------------------------------------------------

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
    Run the full training-set pipeline: load → measure dupes → drop → class-filter → save.

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


# --- Feature engineering -----------------------------------------------------


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
    labels = np.asarray(labels, dtype=str)  # same object→unicode cast as assemble_training_set
    np.savez_compressed(path, X=X, labels=labels, grid=grid)


def build_features(
    train_path: Path = Path("data/openspecy/training_set.npz"),
    out_path: Path = Path("data/openspecy/features.npz"),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the full feature pipeline: load training set → build X → cache → return (X, y).

    Hybrid output: returns (X, y) in-memory for callers AND caches to features.npz
    so training and comparison can np.load without re-running the chain per CV fit.
    """
    data = np.load(train_path)
    wavenumbers, spectra, labels = data["wavenumbers"], data["spectra"], data["labels"]

    print(f"[features] building {len(spectra):,} spectra → {len(WAVENUMBER_GRID)}-d grid …")
    X = build_feature_matrix(wavenumbers, spectra)

    save_feature_matrix(out_path, X, labels, WAVENUMBER_GRID)
    print(f"[features] X={X.shape} {X.dtype}, range {X.min():.3f}–{X.max():.3f}")
    print(f"[features] saved → {out_path}")
    return X, labels


# --- SVM training ------------------------------------------------------------

# RANDOM_STATE is the mechanism that makes the train/test split SHARED across models:
# SVM, RF, and the rule baseline all call load_split()
# with this same seed → all three are evaluated on the identical held-out rows.


def load_split(
    features_path: Path = Path("data/openspecy/features.npz"),
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the cached feature matrix and carve off a locked, stratified test set.

    `stratify=y` makes the 80/20 split preserve each class's proportion, so the
    sparse classes (e.g. polyhaloolefins) appear in both train and test rather
    than landing entirely on one side. The test set is used only for final
    evaluation — never used for tuning.

    Returns X_train, X_test, y_train, y_test.
    """
    data = np.load(features_path)
    X, y = data["X"], data["labels"]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_svm_pipeline(probability: bool = False) -> Pipeline:
    """
    StandardScaler → SVC, as one estimator.

    Bundling them into a Pipeline is what keeps scaling leak-free: when this whole
    object is handed to GridSearchCV, sklearn refits the scaler on each CV fold's
    TRAINING rows only, then applies it to that fold's validation rows. Scaling X
    up front instead would let every fold's validation statistics bleed into the
    scaler the model was trained under.

    StandardScaler standardizes each FEATURE (column = one wavenumber) to zero mean,
    unit variance across samples — distinct from the per-spectrum [0,1] normalization
    already baked into the features. RBF-SVM measures distances in feature space, so
    columns must share a scale or the wide-variance wavenumbers dominate the kernel.
    `class_weight='balanced'` reweights the hinge loss inversely to class frequency.

    `probability=True` (the persisted model only) fits Platt scaling — an
    internal 5-fold CV that calibrates the margins into per-class probabilities, so
    `.predict_proba()` works for the CLI's confidence readout. It's off during tuning
    because it costs an extra CV pass and the grid search never asks for
    probabilities. `random_state` is pinned so that Platt calibration is reproducible.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(class_weight="balanced", probability=probability,
                    random_state=RANDOM_STATE)),
    ])


def tune_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subsample: int = TUNE_SUBSAMPLE,
    random_state: int = RANDOM_STATE,
) -> tuple[dict, float]:
    """
    Grid-search SVM hyperparameters on a stratified SUBSAMPLE of the training set.

    SVC is ~O(n²)–O(n³) in samples, so searching the whole 16k-row train set across
    the grid × folds would be hours. Hyperparameters are stable to sample size, so we
    tune on ~`subsample` stratified rows (minutes) and let the winner carry over to the
    full-data refit. The subsample is drawn with the shared seed for reproducibility.

    Returns (best_params, best_cv_f1_macro).
    """
    # Stratified down-sample: a second train_test_split, but we keep only the
    # "test" side sized to `subsample`, stratified so class proportions hold.
    if subsample < len(X_train):
        _, X_tune, _, y_tune = train_test_split(
            X_train, y_train, test_size=subsample,
            random_state=random_state, stratify=y_train,
        )
    else:
        X_tune, y_tune = X_train, y_train

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        build_svm_pipeline(),
        SVM_PARAM_GRID,
        scoring="f1_macro",   # macro = every class counts equally, imbalance-honest
        cv=cv,
        n_jobs=-1,            # parallelize fits across cores
    )
    search.fit(X_tune, y_tune)
    return search.best_params_, search.best_score_


def train_svm(
    features_path: Path = Path("data/openspecy/features.npz"),
    probability: bool = False,
) -> Pipeline:
    """
    Full SVM pipeline: load shared split → tune on subsample → refit best params
    on the FULL training set → evaluate once on the held-out test set.

    `probability` is passed only to the FINAL refit (persist_model.py turns it on for
    `.predict_proba`); tuning stays cheap with it off. The flag leaves `.predict` unchanged,
    but the argmax of `.predict_proba` disagrees with it on 11 of 4,113 test spectra
    (F1 0.9758 vs 0.9770). The reported F1 and `predict()` both use `.predict`.

    Returns the fitted pipeline (refit on all training data) so it can be persisted.
    """
    X_train, X_test, y_train, y_test = load_split(features_path)
    print(f"[svm] train={X_train.shape[0]:,}  test={X_test.shape[0]:,}")

    best_params, cv_f1 = tune_svm(X_train, y_train)
    print(f"[svm] best params: {best_params}")
    print(f"[svm] tuning CV F1-macro (on subsample): {cv_f1:.3f}")

    # Refit the winning hyperparameters on ALL training rows. set_params takes the
    # same 'svc__C' / 'svc__kernel' keys the grid used, so best_params drops straight in.
    model = build_svm_pipeline(probability=probability).set_params(**best_params)
    model.fit(X_train, y_train)

    # Final evaluation on the held-out test set, which is never used for tuning.
    test_f1 = f1_score(
        y_test, model.predict(X_test), labels=sorted(np.unique(y_train)), average="macro"
    )
    print(f"[svm] held-out test F1-macro: {test_f1:.3f}")
    return model


# --- Random Forest training --------------------------------------------------

# RF reuses the SAME load_split() as SVM (shared seed-42 split) so both models —
# and the rule baseline — are scored on the identical held-out rows. The two
# deviations from train_svm are deliberate: NO Pipeline/scaler, because trees
# split on per-feature thresholds and are scale-invariant; and tuning runs on
# the FULL training set, because RF cost is ~linear and parallel, unlike SVC's ~O(n²–n³) that forced the 4k subsample.


def build_rf(random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    """
    A bare RandomForestClassifier — no Pipeline, no scaler.

    Scaling is pointless for a forest: each tree asks "is feature k above some
    threshold?", and a threshold rescales with its column, so the split is
    identical whether the axis runs 0–1 or 0–1000. `class_weight='balanced'`
    reweights each class inversely to its frequency (same imbalance fix as the SVM).
    `n_jobs=-1` grows the trees across all cores.
    """
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def tune_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[dict, float]:
    """
    Grid-search n_estimators × max_depth on the FULL training set.

    No subsample (unlike tune_svm): RF training is ~O(n·log n·trees) and the fits
    parallelize, so the whole 6-combo × 5-fold grid on all 16k rows is minutes.

    Returns (best_params, best_cv_f1_macro).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        build_rf(random_state),
        RF_PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_params_, search.best_score_


def train_random_forest(
    features_path: Path = Path("data/openspecy/features.npz"),
) -> RandomForestClassifier:
    """
    Full RF pipeline: load shared split → tune on full train → refit best
    params → evaluate once on the held-out test set.

    Mirrors train_svm exactly, minus the subsample step. Returns the fitted forest
    so compare_models can score it head-to-head with the SVM.
    """
    X_train, X_test, y_train, y_test = load_split(features_path)
    print(f"[rf] train={X_train.shape[0]:,}  test={X_test.shape[0]:,}")

    best_params, cv_f1 = tune_rf(X_train, y_train)
    print(f"[rf] best params: {best_params}")
    print(f"[rf] tuning CV F1-macro: {cv_f1:.3f}")

    # set_params with the same keys the grid used → best_params drops straight in.
    model = build_rf().set_params(**best_params)
    model.fit(X_train, y_train)

    # Final evaluation on the held-out test set, which is never used for tuning.
    test_f1 = f1_score(
        y_test, model.predict(X_test), labels=sorted(np.unique(y_train)), average="macro"
    )
    print(f"[rf] held-out test F1-macro: {test_f1:.3f}")
    return model


# --- Model comparison --------------------------------------------------------

# The rule baseline doesn't take a feature vector — it takes a list of peak
# wavenumbers (polymer_rules.classify_by_rules). So before it can vote on the
# test set we bridge each processed 851-d spectrum back into peaks with the
# existing detect_peaks(), then map those to the rule table. That way all
# classifiers start from the IDENTICAL processed spectra and load_split() rows.


def predict_rule_baseline(X: np.ndarray) -> np.ndarray:
    """
    Classify each spectrum in X with the hand-written rule table.

    X is the processed feature matrix (rows are 851-d vectors on WAVENUMBER_GRID).
    For each row we detect peaks on that grid, hand the peak wavenumbers to the
    rule table, and collect the predicted class string. This is the baseline arm
    of the three-way comparison — the interpretable comparator the ML must beat.

    Returns a 1-D array of predicted class labels, length len(X).
    """
    preds = []
    for row in X:
        # detect_peaks expects a normalized [0,1] spectrum + its wavenumber axis,
        # which is exactly what a feature row is. Peak.wavenumber comes straight
        # off WAVENUMBER_GRID, so it lands in the rule table's cm⁻¹ ranges.
        peaks = detect_peaks(row, WAVENUMBER_GRID)
        wavenumbers = [p.wavenumber for p in peaks]
        preds.append(classify_by_rules(wavenumbers))
    return np.array(preds)


def _normed(M: np.ndarray) -> np.ndarray:
    """Row-center and L2-normalize, so a dot product of two rows is their Pearson r."""
    # Pearson r between two vectors == cosine of their mean-subtracted forms.
    # (+1e-12 guards a zero-variance flat row from div-by-0.)
    Mc = M - M.mean(axis=1, keepdims=True)
    return Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-12)


def predict_correlation_baseline(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """
    1-NN by Pearson correlation: each test spectrum takes the label of the most
    correlated training spectrum.

    This is the REAL industry baseline for spectral identification — OpenSpecy
    itself identifies unknowns by correlating them against its reference library.
    Unlike the peak-rule table (a deliberately weak, interpretable comparator),
    a reviewer at an instrument maker recognises correlation matching as *the*
    standard method, so beating-or-matching it is the result that actually counts.

    Both matrices are the processed 851-d feature vectors from load_split(), so
    this baseline sees the IDENTICAL inputs as the SVM/RF — a fair three-way race.

    Returns a 1-D array of predicted class labels, length len(X_test).
    """
    train_n, test_n = _normed(X_train), _normed(X_test)
    sims = test_n @ train_n.T          # (n_test, n_train) correlation matrix
    best = sims.argmax(axis=1)         # most-correlated training row per test row
    return y_train[best]


def compare_models(
    features_path: Path = Path("data/openspecy/features.npz"),
) -> dict:
    """
    Race the rule baseline vs correlation matching vs SVM vs RF on the shared
    held-out test set.

    Trains both ML models (full train_svm/train_random_forest pipelines), then
    scores all four on the IDENTICAL test rows. The fairness hinges on load_split() being
    deterministic: train_svm/train_random_forest each split internally with the
    same seed-42, and the load_split() call here reproduces the exact same rows,
    so every classifier is judged on the same spectra.

    Returns a results dict with the fitted models, each classifier's predictions,
    its F1-macro, and y_test — everything the plotting script needs downstream.
    """
    # Fit the two ML models (each tunes + refits + reports its own held-out F1).
    svm = train_svm(features_path)
    rf = train_random_forest(features_path)

    # Re-derive the SAME train/test rows. Deterministic seed → identical split, so
    # these rows are byte-for-byte the ones the models were evaluated on. X_train
    # is needed too: the correlation baseline matches test spectra against it.
    X_train, X_test, y_train, y_test = load_split(features_path)

    predictions = {
        "rule_baseline": predict_rule_baseline(X_test),
        "correlation_baseline": predict_correlation_baseline(X_train, y_train, X_test),
        "svm": svm.predict(X_test),
        "random_forest": rf.predict(X_test),
    }
    # Score over the 8 real classes only, so the rule baseline's "Unknown"
    # output counts as a miss rather than as a 9th class in the macro average.
    classes = sorted(np.unique(y_train))
    f1_macro = {
        name: f1_score(y_test, y_pred, labels=classes, average="macro")
        for name, y_pred in predictions.items()
    }

    print("\n=== Model comparison (F1-macro, held-out test set) ===")
    for name, score in sorted(f1_macro.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {score:.3f}  {name}")

    return {
        "models": {"svm": svm, "random_forest": rf},
        "predictions": predictions,
        "f1_macro": f1_macro,
        "y_test": y_test,
    }


# --- Persist the winning model -----------------------------------------------

MODEL_PATH = Path("models/svm_classifier.joblib")


def save_model(model: Pipeline, test_f1: float, path: Path = MODEL_PATH) -> Path:
    """
    Dump the fitted model + provenance to one joblib file.

    We save a DICT, not the bare estimator: the model plus enough metadata to know
    what it is months later without re-running anything — when it was trained, the
    score it earned, the class labels it can emit, the feature grid its input must
    match, and the sklearn version it was pickled under (joblib pickles are not
    guaranteed loadable across major sklearn versions, so recording it is the
    breadcrumb that explains a future load failure).

    `model.classes_` is delegated by the Pipeline to the fitted SVC — the alphabetical
    label order the model's outputs follow, so `predict_proba` column k ↔ classes[k].
    """
    path.parent.mkdir(parents=True, exist_ok=True)  # models/ is gitignored; create if absent
    bundle = {
        "model": model,
        "model_type": "svm",
        "trained_at": date.today().isoformat(),
        "test_f1_macro": float(test_f1),
        "classes": list(model.classes_),
        "feature_grid": (GRID_START, GRID_END, GRID_STEP),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, path)
    return path


def load_model(path: Path = MODEL_PATH) -> dict:
    """
    Load the persisted bundle. Returns the dict saved by `save_model`; the fitted
    model is under `["model"]`, ready for `.predict` / `.predict_proba`.
    """
    return joblib.load(path)


# --- Single-spectrum inference -----------------------------------------------

def predict(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    *,
    bundle: dict | None = None,
    model_path: Path = MODEL_PATH,
    top_k: int = 3,
) -> dict:
    """
    Classify one absorbance spectrum: arrays in, prediction out.

    Pure inference helper — takes the raw (wavenumbers, intensities) arrays, NOT a
    file path, so it can be unit-tested on a synthetic spectrum with nothing on disk.

    Input is absorbance, but the model was trained on OpenSpecy's derivative
    library. So the spectrum is first converted to that form
    (`to_openspecy_derivative`: 6 cm⁻¹ resample → Savitzky-Golay first derivative
    → absolute value → min-max), then put through `spectrum_to_features`, the same
    transform the training rows went through.

    `bundle` lets a caller pass an already-loaded model dict (e.g. a test fixture, or
    the CLI loading once); when None we load it from `model_path`.

    The class comes from `model.predict`, the same call the reported F1 scores.
    With Platt scaling it can differ from the highest probability, so `confidence`
    is not always the top of `top_k`.

    Returns
    -------
    dict with keys:
      predicted_class – the label from `model.predict` (str)
      confidence      – that label's probability in [0, 1] (float)
      top_k           – list of {"class", "probability"} dicts, highest first
    """
    if bundle is None:
        bundle = load_model(model_path)
    model, classes = bundle["model"], np.asarray(bundle["classes"])

    # spectrum_to_features returns a 1-D (851,) vector; sklearn wants a 2-D
    # (n_samples, n_features) matrix, so reshape to a single-row batch.
    wn_6cm, derivative = to_openspecy_derivative(wavenumbers, intensities)
    features = spectrum_to_features(wn_6cm, derivative).reshape(1, -1)

    # One Platt-calibrated probability per class (column k ↔ classes[k]); [0] picks
    # the single row back out of the batch.
    probabilities = model.predict_proba(features)[0]

    # argsort is ascending → take the last top_k and flip to get highest-first.
    order = np.argsort(probabilities)[::-1][:top_k]
    ranked = [
        {"class": str(classes[i]), "probability": round(float(probabilities[i]), 4)}
        for i in order
    ]
    predicted_class = str(model.predict(features)[0])
    confidence = probabilities[list(classes).index(predicted_class)]
    return {
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 4),
        "top_k": ranked,
    }
