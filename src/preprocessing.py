"""
Signal preprocessing for IR spectra.

Recommended pipeline order:
  1. convert_to_absorbance  (if data is in % Transmittance)
  2. baseline_correct       (remove sloping / curved baseline)
  3. smooth                 (reduce high-frequency noise)
  4. normalize              (scale to [0, 1] for peak detection)
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def convert_to_absorbance(transmittance: np.ndarray) -> np.ndarray:
    """
    Convert % Transmittance to Absorbance: A = -log10(T/100).

    Clips transmittance to (0, 100] to avoid log(0).
    """
    t = np.clip(transmittance, 1e-6, 100.0)
    return -np.log10(t / 100.0)


def baseline_als(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 10,
) -> np.ndarray:
    """
    Asymmetric Least Squares (ALS) baseline estimation.

    Fits a smooth baseline that lies below the spectrum by iteratively
    weighting residuals: peaks get low weight, baseline regions get high
    weight.

    Parameters
    ----------
    y     : input spectrum (absorbance)
    lam   : smoothness of baseline (larger = smoother); try 1e4–1e7
    p     : asymmetry parameter (fraction of points treated as baseline);
            use small values (~0.01–0.05) for spectra with many peaks
    n_iter: number of reweighting iterations (10 is usually enough)

    Returns
    -------
    baseline : estimated baseline array, same length as y
    """
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    H = lam * D.T.dot(D)
    w = np.ones(L)

    for _ in range(n_iter):
        W = diags(w, 0)
        Z = W + H
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)

    return z


def baseline_correct(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
) -> np.ndarray:
    """
    Subtract the ALS baseline from the spectrum.

    Returns the baseline-corrected signal (negative values clipped to 0).
    """
    baseline = baseline_als(y, lam=lam, p=p)
    corrected = y - baseline
    return np.clip(corrected, 0, None)


def smooth(
    signal: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Savitzky-Golay smoothing filter.

    Preferred over simple moving average for spectra because it preserves
    peak positions and shapes while suppressing high-frequency noise.

    Parameters
    ----------
    window_length : number of data points in the filter window (must be odd)
    polyorder     : degree of the fitting polynomial (must be < window_length)
    """
    wl = window_length if window_length % 2 == 1 else window_length + 1
    min_wl = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    wl = max(wl, min_wl)
    # Clamp to signal length (must be odd and <= len(signal))
    max_wl = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    wl = min(wl, max(max_wl, min_wl))
    return savgol_filter(signal, window_length=wl, polyorder=polyorder)


def normalize(intensity: np.ndarray) -> np.ndarray:
    """
    Min-max normalization to [0, 1].

    If the range is zero (flat signal), the array is returned unchanged.
    """
    min_val = np.min(intensity)
    max_val = np.max(intensity)
    span = max_val - min_val

    if span == 0:
        return intensity

    return (intensity - min_val) / span
