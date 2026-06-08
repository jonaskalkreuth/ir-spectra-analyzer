"""
Download and parse the OpenSpecy FTIR reference library.

Source: OSF DOI 10.17605/OSF.IO/X7DPZ — derivative-processed spectra.
The library contains thousands of reference FTIR (and Raman) spectra with
polymer-class labels. We keep FTIR only and return tidy
(wavenumbers, intensities, label) records for model training.

Data flow
---------
1. download_openspecy()   → fetches derivative.rds from OSF
2. export_to_csv()        → runs an R script to convert .rds → three CSVs
                            (wavenumbers.csv, spectra.csv, metadata.csv)
3. load_openspecy()       → reads the CSVs into numpy arrays ready for sklearn

The R export step exists because derivative.rds uses an OpenSpecy R6 class
that no pure-Python parser can handle efficiently. The CSVs are the durable
artefact; the .rds is only needed once.
"""

import subprocess
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

# x7dpz is the OSF *project* node; 2qbkt is the file-level GUID for derivative.rds.
# OSF project /download returns 500; file GUID /download works correctly.
_OSF_URL = "https://osf.io/2qbkt/download"
_DEFAULT_RDS = Path("data/openspecy/derivative.rds")
_DEFAULT_DIR = Path("data/openspecy")

# Minimal R script that extracts FTIR spectra from the OpenSpecy object.
# Written as a string so the pipeline is self-contained — no separate .R file needed.
_EXPORT_R_SCRIPT = """\
args    <- commandArgs(trailingOnly = TRUE)
rds     <- args[1]
out_dir <- args[2]

cat("[R] Loading", rds, "\\n")
obj <- readRDS(rds)

is_ftir  <- obj$metadata$spectrum_type == "ftir"
ftir_idx <- which(is_ftir)

cat("[R] FTIR spectra:", length(ftir_idx), "of", nrow(obj$metadata), "total\\n")

# Metadata: only the two columns we use downstream
meta <- obj$metadata[ftir_idx, c("sample_name", "material_class")]
write.csv(meta, file.path(out_dir, "metadata.csv"), row.names = FALSE)

# Shared wavenumber axis
write.csv(
  data.frame(wavenumber = obj$wavenumber),
  file.path(out_dir, "wavenumbers.csv"),
  row.names = FALSE
)

# Spectra matrix transposed to (n_spectra x n_wavenumbers)
cat("[R] Transposing spectra matrix...\\n")
mat <- t(as.matrix(obj$spectra[, ftir_idx]))
colnames(mat) <- round(obj$wavenumber, 1)
write.csv(mat, file.path(out_dir, "spectra.csv"), row.names = FALSE)

cat("[R] Export complete.\\n")
"""


def download_openspecy(
    dest: Path = _DEFAULT_RDS,
    *,
    force: bool = False,
) -> Path:
    """
    Download derivative.rds from OSF to *dest*.

    Skips the download if the file already exists (pass force=True to re-download).
    Raises RuntimeError immediately if the download fails — fail loud, fail early.

    Parameters
    ----------
    dest  : local path to save the file (parent dirs created automatically)
    force : re-download even if *dest* already exists

    Returns
    -------
    Path to the downloaded file.
    """
    dest = Path(dest)

    if dest.exists() and not force:
        print(f"[data_acquisition] Already present: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data_acquisition] Downloading from {_OSF_URL} ...")

    try:
        urllib.request.urlretrieve(_OSF_URL, dest)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Download failed — check your internet connection.\n"
            f"URL: {_OSF_URL}\n"
            f"Error: {exc}"
        ) from exc

    size_mb = dest.stat().st_size / 1_000_000
    print(f"[data_acquisition] Saved {size_mb:.1f} MB → {dest}")
    return dest


def export_to_csv(
    rds_path: Path = _DEFAULT_RDS,
    out_dir: Path = _DEFAULT_DIR,
    *,
    rscript: str = "Rscript",
    force: bool = False,
) -> None:
    """
    Convert derivative.rds to three CSVs via a minimal R script.

    Skips the export if wavenumbers.csv already exists (pass force=True to redo).
    Raises RuntimeError if Rscript is not found or the script fails.

    Outputs (in out_dir)
    --------------------
    Three files instead of one because the data has three different shapes:

    wavenumbers.csv   : (n_wavenumbers,)          — the shared x-axis (one vector,
                        same for every spectrum)
    spectra.csv       : (n_spectra, n_wavenumbers) — intensity matrix in WIDE format:
                        one row per spectrum, one column per wavenumber point.
                        Wide format is chosen over long format (one row per data point)
                        because it loads directly as the feature matrix X for sklearn
                        without any reshaping. Long format would be ~43 million rows.
    metadata.csv      : (n_spectra, 2)             — sample_name + material_class label
                        (the y vector). Separate from spectra.csv because it has a
                        different number of columns.
    """
    out_dir  = Path(out_dir)
    wn_csv   = out_dir / "wavenumbers.csv"

    sp_csv = out_dir / "spectra.csv"
    if wn_csv.exists() and sp_csv.exists() and not force:
        print(f"[data_acquisition] CSVs already present in {out_dir}, skipping export.")
        return

    # Write the R script to a temp file
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(_EXPORT_R_SCRIPT)
        script_path = f.name

    try:
        result = subprocess.run(
            [rscript, script_path, str(rds_path), str(out_dir)],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                f"R export failed (exit {result.returncode}):\n{result.stderr}"
            )
    except FileNotFoundError:
        raise RuntimeError(
            f"Rscript not found at '{rscript}'. "
            "Install R from https://cran.r-project.org and ensure it is on PATH."
        ) from None
    finally:
        os.unlink(script_path)


def load_openspecy(
    data_dir: Path = _DEFAULT_DIR,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Load the exported CSVs into numpy arrays ready for scikit-learn.

    Returns
    -------
    wavenumbers : shape (n_wavenumbers,)   — shared x-axis in cm⁻¹
    spectra     : shape (n_spectra, n_wavenumbers) — intensity matrix, float32
    labels      : pd.Series of length n_spectra   — material_class strings
    """
    data_dir = Path(data_dir)
    wn_csv   = data_dir / "wavenumbers.csv"
    sp_csv   = data_dir / "spectra.csv"
    meta_csv = data_dir / "metadata.csv"

    for p in (wn_csv, sp_csv, meta_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "Run export_to_csv() first to generate the CSVs from derivative.rds."
            )

    print("[data_acquisition] Loading wavenumbers...")
    wavenumbers = pd.read_csv(wn_csv)["wavenumber"].to_numpy(dtype=np.float64)

    print("[data_acquisition] Loading spectra matrix (this takes ~30 s)...")
    # float32 halves memory vs float64; precision is fine for IR intensities
    spectra = pd.read_csv(sp_csv).to_numpy(dtype=np.float32)

    print("[data_acquisition] Loading metadata...")
    metadata = pd.read_csv(meta_csv)
    labels   = metadata["material_class"]

    print(
        f"[data_acquisition] Loaded {spectra.shape[0]:,} spectra × "
        f"{spectra.shape[1]:,} wavenumber points. "
        f"Label classes: {labels.nunique()}"
    )

    return wavenumbers, spectra, labels
