# Preprocessing defaults
SAVGOL_WINDOW = 11          # Savitzky-Golay window length (must be odd)
SAVGOL_POLYORDER = 3        # Savitzky-Golay polynomial order

ALS_LAMBDA = 1e5            # ALS baseline smoothness (1e4–1e7)
ALS_P = 0.01                # ALS asymmetry (0.01–0.05 for typical spectra)

# Peak detection defaults
MIN_PROMINENCE = 0.05       # minimum peak prominence (normalized scale 0–1)
MIN_HEIGHT = 0.02           # minimum peak intensity (normalized scale 0–1)
MIN_DISTANCE_CM = 20.0      # minimum separation between peaks in cm⁻¹

# Functional group assignment
MIN_INTENSITY_FILTER = "medium"   # filter out 'weak' / 'variable' hits by default

# Wavenumber region labels (for reporting)
REGIONS = {
    "X-H stretch":    (2500, 4000),
    "Triple bond":    (2000, 2500),
    "Double bond":    (1500, 2000),
    "Fingerprint":    (400,  1500),
}

# Paths
DEFAULT_DATA_PATH = "data/sample_spectra.csv"
DEFAULT_OUTPUT_PATH = "results.json"
