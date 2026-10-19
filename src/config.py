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

# ML / classifier training
RANDOM_STATE = 42           # shared seed → SVM/RF/baseline get the identical train/test split
TEST_SIZE = 0.2             # fraction held out for the one-shot final evaluation
TUNE_SUBSAMPLE = 4000       # stratified rows used for the hyperparameter search only

# SVM grid: linear vs RBF (the kernel comparison), C over 4 decades, gamma RBF-only.
# List-of-dicts (not one dict) so gamma is never paired with the linear kernel.
SVM_PARAM_GRID = [
    {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10, 100]},
    {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10, 100],
     "svc__gamma": ["scale", 0.01, 0.001]},
]

# RF grid: the two knobs the plan tunes — forest size and tree depth.
# Plain dict (no kernel-style incompatibilities to guard against, unlike SVM).
# Keys have NO prefix because RF is a bare estimator, not a Pipeline.
#   n_estimators: more trees = steadier vote, diminishing returns; 200/400 brackets it.
#   max_depth:    None = grow until pure (max variance); 20/40 cap depth to curb overfit.
RF_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20, 40],
}

# Paths
DEFAULT_DATA_PATH = "data/sample_spectra.csv"
