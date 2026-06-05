import numpy as np
import pytest
from src.peak_detection import detect_peaks, Peak


def _gaussian(x, center, sigma, height):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def make_spectrum(peak_positions, peak_heights, n=500):
    """Synthetic spectrum: sum of Gaussians on a flat baseline."""
    wn = np.linspace(4000, 400, n)
    signal = np.zeros(n)
    for pos, h in zip(peak_positions, peak_heights):
        signal += _gaussian(wn, pos, 30, h)
    return wn, signal


def test_detects_single_peak():
    wn, sig = make_spectrum([1700], [0.8])
    peaks = detect_peaks(sig, wn)
    assert len(peaks) == 1
    assert abs(peaks[0].wavenumber - 1700) < 30


def test_detects_multiple_peaks():
    wn, sig = make_spectrum([3400, 2900, 1720], [0.9, 0.7, 0.95])
    peaks = detect_peaks(sig, wn)
    assert len(peaks) == 3


def test_returns_peak_objects():
    wn, sig = make_spectrum([1700], [0.8])
    peaks = detect_peaks(sig / sig.max(), wn)
    p = peaks[0]
    assert isinstance(p, Peak)
    assert p.fwhm_cm > 0
    assert p.prominence > 0


def test_sorted_descending():
    wn, sig = make_spectrum([1000, 2000, 3000], [0.5, 0.6, 0.7])
    peaks = detect_peaks(sig, wn)
    wns = [p.wavenumber for p in peaks]
    assert wns == sorted(wns, reverse=True)


def test_min_distance_filters_close_peaks():
    wn, sig = make_spectrum([1700, 1720], [0.8, 0.8])
    peaks_far = detect_peaks(sig, wn, min_distance_cm=50)
    peaks_close = detect_peaks(sig, wn, min_distance_cm=5)
    # With 50 cm-1 minimum distance only one of the two close peaks survives
    assert len(peaks_far) <= len(peaks_close)


def test_flat_signal_returns_no_peaks():
    wn = np.linspace(4000, 400, 200)
    sig = np.full(200, 0.1)
    peaks = detect_peaks(sig, wn)
    assert len(peaks) == 0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        detect_peaks(np.ones(100), np.ones(50))
