import numpy as np
import pytest

from src.classifier import (
    WAVENUMBER_GRID,
    build_svm_pipeline,
    count_exact_duplicates,
    drop_exact_duplicates,
    filter_classes,
    load_model,
    predict,
    predict_correlation_baseline,
    resample_to_grid,
    save_model,
    spectrum_to_features,
)


def _gaussian(x, center, sigma, height):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def make_spectrum(centers, heights, wn):
    """Sum of Gaussian peaks sampled on the given wavenumber axis."""
    signal = np.zeros_like(wn, dtype=float)
    for c, h in zip(centers, heights):
        signal += _gaussian(wn, c, 30, h)
    return signal


# --- Cluster 2: pure transforms ----------------------------------------------

def test_resample_lands_on_the_grid():
    # Native axis runs DESCENDING (4000→400, as the OpenSpecy/SDBS files do) to
    # prove resample sorts before interpolating; np.interp would error otherwise.
    wn = np.linspace(4000, 400, 600)
    sig = make_spectrum([1700], [0.8], wn)
    out = resample_to_grid(wn, sig)
    assert out.shape == WAVENUMBER_GRID.shape  # always 851, regardless of input length


def test_resample_drops_nan_padding():
    # OpenSpecy pads unmeasured regions with NaN; np.interp would smear those
    # into the output, so resample must strip them first.
    wn = np.linspace(400, 4000, 600)
    sig = make_spectrum([1700], [0.8], wn)
    sig[:50] = np.nan
    out = resample_to_grid(wn, sig)
    assert not np.isnan(out).any()


def test_features_are_normalized_to_unit_interval():
    # spectrum_to_features ends on min-max normalize, so every feature vector
    # the models ever see is bounded to [0, 1] no matter the input scale.
    wn = np.linspace(400, 4000, 600)
    sig = make_spectrum([1700, 2900], [50.0, 30.0], wn)  # deliberately un-normalized
    feats = spectrum_to_features(wn, sig)
    assert feats.shape == WAVENUMBER_GRID.shape
    assert feats.min() >= 0.0 and feats.max() <= 1.0


# --- Cluster 3: training-set assembly helpers --------------------------------

def test_counts_exact_duplicate_rows():
    # Rows 0 and 2 are identical; only row 2 is the repeat → count is 1
    # (keep="first" never flags the first occurrence).
    spectra = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [1.0, 2.0, 3.0],
    ])
    assert count_exact_duplicates(spectra) == 1


def test_duplicate_detection_treats_nan_pad_as_equal():
    # Two spectra padded with NaN in the SAME columns must count as duplicates.
    # Plain numpy ==  says NaN != NaN, so this only works because the helper
    # hashes rows via pandas, which treats matching NaNs as equal.
    spectra = np.array([
        [np.nan, 2.0, 3.0],
        [np.nan, 2.0, 3.0],
    ])
    assert count_exact_duplicates(spectra) == 1


def test_drop_duplicates_keeps_labels_aligned():
    # The duplicate row (index 2) is dropped; the SURVIVING rows must keep
    # their original labels — the load-bearing invariant of masking two arrays
    # in lockstep.
    spectra = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [1.0, 1.0],
    ])
    labels = np.array(["PE", "PP", "PE"])
    kept_spectra, kept_labels = drop_exact_duplicates(spectra, labels)
    assert kept_spectra.shape[0] == 2
    assert list(kept_labels) == ["PE", "PP"]


def test_filter_classes_drops_rare_tail():
    # "rare" appears once, "common" three times; min_count=2 cuts the tail
    # while leaving every "common" sample intact (no down-sampling).
    spectra = np.arange(8).reshape(4, 2).astype(float)
    labels = np.array(["common", "common", "common", "rare"])
    kept_spectra, kept_labels = filter_classes(spectra, labels, min_count=2)
    assert set(kept_labels) == {"common"}
    assert kept_spectra.shape[0] == 3


# --- Cluster 3b: correlation library-matching baseline -----------------------

def test_correlation_baseline_matches_nearest_class():
    # Two well-separated class prototypes (peak at 1700 vs 2900). Each test
    # spectrum is a noisy copy of one prototype → 1-NN correlation must return
    # that prototype's label.
    wn = WAVENUMBER_GRID
    proto_a = make_spectrum([1700], [1.0], wn)
    proto_b = make_spectrum([2900], [1.0], wn)
    X_train = np.stack([proto_a, proto_b])
    y_train = np.array(["A", "B"])

    rng = np.random.default_rng(0)
    X_test = np.stack([
        proto_b + rng.normal(0, 0.01, wn.shape),   # closest to B
        proto_a + rng.normal(0, 0.01, wn.shape),   # closest to A
    ])
    preds = predict_correlation_baseline(X_train, y_train, X_test)
    assert list(preds) == ["B", "A"]


def test_correlation_baseline_is_scale_invariant():
    # Pearson correlation ignores amplitude, so a test spectrum that is a
    # same-shape but 10x-taller copy of a training row still matches its label.
    wn = WAVENUMBER_GRID
    proto_a = make_spectrum([1700], [1.0], wn)
    proto_b = make_spectrum([2900], [1.0], wn)
    X_train = np.stack([proto_a, proto_b])
    y_train = np.array(["A", "B"])

    preds = predict_correlation_baseline(X_train, y_train, np.stack([proto_a * 10.0]))
    assert preds[0] == "A"


# --- Cluster 4: model lifecycle (save/load + predict) ------------------------

# Three synthetic classes, each a single Gaussian at a distinct, well-separated
# wavenumber → trivially learnable, so a real SVM trains in milliseconds.
_CLASS_CENTERS = {"alpha": 1700, "beta": 2900, "gamma": 3400}


def synth_spectrum(center, seed=None):
    """A raw (wavenumbers, intensities) spectrum: one jittered Gaussian + noise."""
    rng = np.random.default_rng(seed)
    wn = np.linspace(400, 4000, 700)  # native axis ≠ the 851 grid → forces a resample
    sig = make_spectrum([center + rng.normal(0, 5)], [1.0], wn)
    sig += rng.normal(0, 0.01, sig.shape)
    return wn, sig


@pytest.fixture(scope="module")
def tiny_model():
    """A real SVM fit on ~36 synthetic spectra (Option A: self-contained).

    Needs no data files and no models/*.joblib, so the suite runs on a fresh
    clone. probability=True mirrors the persisted model so predict()'s
    predict_proba path is exercised. Module-scoped: fit once, reused by all tests.
    """
    X_rows, y = [], []
    for label, center in _CLASS_CENTERS.items():
        for i in range(12):
            wn, sig = synth_spectrum(center, seed=i)
            X_rows.append(spectrum_to_features(wn, sig))
            y.append(label)
    return build_svm_pipeline(probability=True).fit(np.array(X_rows), np.array(y))


def _bundle(model):
    """Wrap a fitted model the way save_model would, for in-memory predict()."""
    return {"model": model, "classes": list(model.classes_)}


def test_predict_recovers_the_class(tiny_model):
    # A fresh beta-class spectrum (peak ~2900) must come back as "beta" — the
    # end-to-end behaviour: raw arrays → spectrum_to_features → model → label.
    wn, sig = synth_spectrum(_CLASS_CENTERS["beta"], seed=99)
    result = predict(wn, sig, bundle=_bundle(tiny_model))
    assert result["predicted_class"] == "beta"


def test_predict_output_is_well_formed(tiny_model):
    wn, sig = synth_spectrum(_CLASS_CENTERS["alpha"], seed=7)
    result = predict(wn, sig, bundle=_bundle(tiny_model), top_k=3)
    assert 0.0 <= result["confidence"] <= 1.0
    probs = [r["probability"] for r in result["top_k"]]
    assert len(probs) == 3
    assert probs == sorted(probs, reverse=True)         # highest first
    assert result["confidence"] == probs[0]              # winner == top of the ranking


def test_save_load_roundtrip_predicts_identically(tiny_model, tmp_path):
    # Persist → reload → the reloaded model must give the SAME prediction as the
    # in-memory one. tmp_path keeps the test off the real models/ dir.
    path = save_model(tiny_model, test_f1=0.95, path=tmp_path / "svm.joblib")
    reloaded = load_model(path)

    wn, sig = synth_spectrum(_CLASS_CENTERS["gamma"], seed=3)
    from_disk = predict(wn, sig, bundle=reloaded)
    in_memory = predict(wn, sig, bundle=_bundle(tiny_model))
    assert from_disk["predicted_class"] == in_memory["predicted_class"]
    assert from_disk["confidence"] == in_memory["confidence"]


class _StubModel:
    """A fake estimator with fixed probabilities — lets us assert the EXACT
    top_k ordering without depending on what a trained model happens to output."""
    classes_ = np.array(["A", "B", "C"])

    def predict_proba(self, X):
        return np.array([[0.1, 0.7, 0.2]])  # B highest, then C, then A


def test_top_k_is_ranked_highest_first():
    wn, sig = synth_spectrum(1700, seed=1)  # spectrum is irrelevant; proba is fixed
    result = predict(wn, sig, bundle={"model": _StubModel(), "classes": ["A", "B", "C"]})
    assert result["predicted_class"] == "B"
    assert result["confidence"] == 0.7
    assert [r["class"] for r in result["top_k"]] == ["B", "C", "A"]
