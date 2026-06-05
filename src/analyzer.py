"""
Main IR analysis pipeline.

Usage
-----
from src.analyzer import analyze

result = analyze("data/my_spectrum.csv")
"""

import logging
import numpy as np

from .loader import load_spectrum
from .preprocessing import baseline_correct, smooth, normalize
from .peak_detection import detect_peaks
from .functional_groups import assign_groups, summarize_compound_classes
from . import config

logger = logging.getLogger(__name__)


def _label_region(wavenumber: float) -> str:
    for label, (lo, hi) in config.REGIONS.items():
        if lo <= wavenumber <= hi:
            return label
    return "Out of range"


def analyze(
    path: str,
    *,
    correct_baseline: bool = True,
    input_is_transmittance: bool = False,
    min_prominence: float = config.MIN_PROMINENCE,
    min_height: float = config.MIN_HEIGHT,
    min_distance_cm: float = config.MIN_DISTANCE_CM,
    min_intensity_filter: str = config.MIN_INTENSITY_FILTER,
) -> dict:
    """
    Full IR analysis pipeline.

    Steps
    -----
    1. Load CSV spectrum
    2. (Optional) Convert % Transmittance → Absorbance
    3. (Optional) Baseline correction (ALS)
    4. Savitzky-Golay smoothing
    5. Min-max normalization
    6. Prominence-based peak detection
    7. Functional group assignment

    Parameters
    ----------
    path :
        Path to the CSV spectrum file.
    correct_baseline :
        Apply ALS baseline correction (recommended for raw spectra).
    input_is_transmittance :
        Set True if the intensity column is % Transmittance instead of
        Absorbance.
    min_prominence, min_height, min_distance_cm :
        Peak detection thresholds (see peak_detection.detect_peaks).
    min_intensity_filter :
        Minimum expected band intensity for functional group matches
        ('strong', 'medium', 'weak').

    Returns
    -------
    dict with keys:
      source          – file path
      num_peaks       – number of peaks detected
      peaks           – list of peak dicts (wavenumber, intensity,
                        prominence, fwhm_cm, region)
      groups          – list of functional group match dicts
      compound_classes – list of compound classes ordered by frequency
    """
    logger.info("Loading spectrum: %s", path)
    df = load_spectrum(path)

    wavenumbers = df["wavenumber"].values.astype(float)
    intensity = df["intensity"].values.astype(float)

    # Ensure spectrum runs high→low wavenumber (conventional IR orientation)
    if wavenumbers[0] < wavenumbers[-1]:
        wavenumbers = wavenumbers[::-1]
        intensity = intensity[::-1]

    if input_is_transmittance:
        logger.debug("Converting transmittance to absorbance.")
        from .preprocessing import convert_to_absorbance
        intensity = convert_to_absorbance(intensity)

    if correct_baseline:
        logger.debug("Applying ALS baseline correction.")
        intensity = baseline_correct(
            intensity, lam=config.ALS_LAMBDA, p=config.ALS_P
        )

    logger.debug("Smoothing with Savitzky-Golay (window=%d, order=%d).",
                 config.SAVGOL_WINDOW, config.SAVGOL_POLYORDER)
    intensity = smooth(
        intensity,
        window_length=config.SAVGOL_WINDOW,
        polyorder=config.SAVGOL_POLYORDER,
    )

    intensity = normalize(intensity)

    logger.debug("Detecting peaks.")
    raw_peaks = detect_peaks(
        intensity,
        wavenumbers,
        min_prominence=min_prominence,
        min_height=min_height,
        min_distance_cm=min_distance_cm,
    )

    peak_wavenumbers = [p.wavenumber for p in raw_peaks]

    logger.debug("Assigning functional groups to %d peaks.", len(raw_peaks))
    groups = assign_groups(peak_wavenumbers, min_intensity=min_intensity_filter)
    compound_classes = summarize_compound_classes(groups)

    peak_dicts = [
        {
            "wavenumber": round(p.wavenumber, 1),
            "intensity": round(p.intensity, 4),
            "prominence": round(p.prominence, 4),
            "fwhm_cm": round(p.fwhm_cm, 1),
            "region": _label_region(p.wavenumber),
        }
        for p in raw_peaks
    ]

    logger.info(
        "Analysis complete: %d peaks, %d functional group matches, "
        "%d compound classes.",
        len(raw_peaks), len(groups), len(compound_classes),
    )

    return {
        "source": str(path),
        "num_peaks": len(raw_peaks),
        "peaks": peak_dicts,
        "groups": groups,
        "compound_classes": compound_classes,
    }
