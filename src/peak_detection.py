"""
Peak detection for IR spectra.

Uses scipy.signal.find_peaks with prominence-based filtering, which is more
robust than height-only detection for spectra with varying baselines.

Peak widths are calculated as FWHM (Full Width at Half Maximum) in data-point
units; the caller can convert to cm⁻¹ by multiplying by the wavenumber spacing.
"""

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from dataclasses import dataclass

from . import config


@dataclass
class Peak:
    index: int
    wavenumber: float
    intensity: float
    prominence: float
    fwhm_pts: float     # FWHM in data-point units
    fwhm_cm: float      # FWHM in cm⁻¹ (requires wavenumber array)


def detect_peaks(
    intensity: np.ndarray,
    wavenumbers: np.ndarray,
    min_prominence: float = config.MIN_PROMINENCE,
    min_height: float = config.MIN_HEIGHT,
    min_distance_cm: float = config.MIN_DISTANCE_CM,
) -> list:
    """
    Detect absorption peaks in a preprocessed (normalized) IR spectrum.

    Parameters
    ----------
    intensity :
        Normalized absorbance array (0–1).
    wavenumbers :
        Corresponding wavenumber array in cm⁻¹ (same length as intensity,
        may be in either ascending or descending order).
    min_prominence :
        Minimum peak prominence relative to surrounding signal.
        Filters out shoulders and noise bumps.
    min_height :
        Minimum absolute intensity value at the peak.
    min_distance_cm :
        Minimum separation between neighbouring peaks in cm⁻¹.

    Returns
    -------
    List of Peak objects, sorted by wavenumber descending.
    """
    if len(intensity) != len(wavenumbers):
        raise ValueError(
            f"intensity ({len(intensity)}) and wavenumbers ({len(wavenumbers)}) "
            "must have the same length."
        )

    # Estimate average wavenumber spacing (always positive)
    spacing = abs(float(np.median(np.diff(wavenumbers))))
    if spacing == 0:
        spacing = 1.0

    min_distance_pts = max(1, int(round(min_distance_cm / spacing)))

    indices, _ = find_peaks(
        intensity,
        height=min_height,
        distance=min_distance_pts,
    )

    if len(indices) == 0:
        return []

    prominences, _, _ = peak_prominences(intensity, indices)
    keep = prominences >= min_prominence
    indices = indices[keep]
    prominences = prominences[keep]

    # FWHM via scipy (rel_height=0.5 → half maximum)
    widths, _, _, _ = peak_widths(intensity, indices, rel_height=0.5)

    peaks = []
    for i, idx in enumerate(indices):
        fwhm_cm = float(widths[i]) * spacing
        peaks.append(
            Peak(
                index=int(idx),
                wavenumber=float(wavenumbers[idx]),
                intensity=float(intensity[idx]),
                prominence=float(prominences[i]),
                fwhm_pts=float(widths[i]),
                fwhm_cm=fwhm_cm,
            )
        )

    # Sort by wavenumber descending (conventional IR presentation)
    peaks.sort(key=lambda p: p.wavenumber, reverse=True)
    return peaks
