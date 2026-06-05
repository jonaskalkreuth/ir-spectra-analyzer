# Polymer classifier: SVM + Random Forest trained on OpenSpecy FTIR spectra.
# Feature vector: spectrum resampled to fixed 400–4000 cm⁻¹ grid at 4 cm⁻¹
# steps (~900 features), normalized intensities used directly.
# TODO: implemented in steps B1b-1c-i through B1b-1c-v.

import numpy as np


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
