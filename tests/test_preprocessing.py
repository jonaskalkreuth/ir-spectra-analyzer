import numpy as np
import pytest
from src.preprocessing import (
    normalize,
    smooth,
    baseline_correct,
    convert_to_absorbance,
)


def test_normalize_range():
    sig = np.array([1.0, 5.0, 3.0, 2.0])
    out = normalize(sig)
    assert pytest.approx(out.min()) == 0.0
    assert pytest.approx(out.max()) == 1.0


def test_normalize_flat_signal():
    sig = np.full(10, 0.5)
    out = normalize(sig)
    np.testing.assert_array_equal(out, sig)


def test_smooth_preserves_length():
    sig = np.random.rand(200)
    out = smooth(sig)
    assert len(out) == len(sig)


def test_smooth_reduces_noise():
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.5, 300)
    smooth_noise = smooth(noise)
    assert smooth_noise.std() < noise.std()


def test_baseline_correct_non_negative():
    sig = np.abs(np.random.default_rng(0).normal(0.5, 0.2, 200))
    out = baseline_correct(sig)
    assert (out >= 0).all()


def test_baseline_correct_removes_slope():
    x = np.linspace(0, 1, 300)
    baseline = 0.5 * x          # linearly rising baseline
    peak = 0.8 * np.exp(-((x - 0.5) / 0.05) ** 2)
    sig = baseline + peak
    corrected = baseline_correct(sig)
    # After correction the signal should be flatter away from the peak
    # Check that the slope outside the peak region is reduced
    flank = corrected[:50]
    assert flank.std() < sig[:50].std()


def test_convert_to_absorbance_100pct():
    # 100 % T → absorbance = 0
    result = convert_to_absorbance(np.array([100.0]))
    assert pytest.approx(result[0], abs=1e-6) == 0.0


def test_convert_to_absorbance_10pct():
    # 10 % T → absorbance = 1.0
    result = convert_to_absorbance(np.array([10.0]))
    assert pytest.approx(result[0], abs=1e-4) == 1.0


def test_convert_to_absorbance_clamps_zero():
    # Should not crash on zero transmittance
    result = convert_to_absorbance(np.array([0.0]))
    assert np.isfinite(result[0])
