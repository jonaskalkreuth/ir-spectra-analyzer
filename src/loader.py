"""
Spectrum loader for CSV-based IR data files.

Expected CSV format (header row required):
  wavenumber,intensity
  4000,0.05
  3990,0.06
  ...

Column names are case-insensitive and common aliases are accepted:
  wavenumber  → 'wavenumber', 'wavenumbers', 'wn', 'cm-1', 'cm_1', 'freq'
  intensity   → 'intensity', 'absorbance', 'transmittance', 'abs', 'trans', 'signal'
"""

import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Tuples, not sets: the order is the match priority, so the column choice is deterministic.
_WN_ALIASES = ("wavenumber", "wavenumbers", "wn", "cm-1", "cm_1", "freq")
_INT_ALIASES = ("intensity", "absorbance", "transmittance", "abs", "trans", "signal")


def _find_column(columns: list, aliases: tuple) -> str | None:
    """Return the column matching the highest-priority alias (case-insensitive)."""
    lower_map = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def load_spectrum(path: str) -> pd.DataFrame:
    """
    Load a single IR spectrum from a CSV file.

    Returns a DataFrame with exactly two columns:
      - 'wavenumber': float, cm⁻¹
      - 'intensity':  float, absorbance or transmittance units

    Raises
    ------
    FileNotFoundError  : if the file does not exist
    ValueError         : if required columns are missing or data is invalid
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Spectrum file not found: {path}")
    if p.stat().st_size == 0:
        raise ValueError(f"Spectrum file is empty: {path}")

    try:
        df = pd.read_csv(p)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV file '{path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Spectrum file contains no data rows: {path}")

    # Normalize column names
    wn_col = _find_column(df.columns.tolist(), _WN_ALIASES)
    int_col = _find_column(df.columns.tolist(), _INT_ALIASES)

    if wn_col is None:
        raise ValueError(
            f"Could not find a wavenumber column in '{path}'. "
            f"Expected one of: {sorted(_WN_ALIASES)}. "
            f"Found: {df.columns.tolist()}"
        )
    if int_col is None:
        raise ValueError(
            f"Could not find an intensity column in '{path}'. "
            f"Expected one of: {sorted(_INT_ALIASES)}. "
            f"Found: {df.columns.tolist()}"
        )

    df = df[[wn_col, int_col]].rename(
        columns={wn_col: "wavenumber", int_col: "intensity"}
    )

    # Drop rows where either column is missing
    n_before = len(df)
    df = df.dropna(subset=["wavenumber", "intensity"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped} row(s) with missing values from '{path}'.",
            stacklevel=2,
        )

    if df.empty:
        raise ValueError(f"No valid data rows after removing NaNs in '{path}'.")

    # Coerce to numeric (catches non-numeric strings)
    for col in ("wavenumber", "intensity"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_bad = df[["wavenumber", "intensity"]].isna().any(axis=1).sum()
    if n_bad > 0:
        df = df.dropna(subset=["wavenumber", "intensity"])
        warnings.warn(
            f"Dropped {n_bad} non-numeric row(s) from '{path}'.",
            stacklevel=2,
        )
        if df.empty:
            raise ValueError(f"No numeric data rows in '{path}'.")

    # Validate wavenumber range (IR: ~400–4000 cm⁻¹; allow some slack)
    wn = df["wavenumber"]
    if wn.max() < 200 or wn.min() > 8000:
        warnings.warn(
            f"Wavenumber range {wn.min():.0f}–{wn.max():.0f} cm⁻¹ looks unusual "
            f"for mid-IR (expected ~400–4000 cm⁻¹).",
            stacklevel=2,
        )

    df = df.reset_index(drop=True)
    return df


def load_multiple(paths: list) -> pd.DataFrame:
    """
    Load multiple spectra and return them as a single tidy DataFrame.

    Adds a 'source' column with the originating file path so spectra can
    be distinguished after concatenation.
    """
    frames = []
    for p in paths:
        df = load_spectrum(p)
        df["source"] = str(p)
        frames.append(df)

    if not frames:
        raise ValueError("No paths provided to load_multiple.")

    return pd.concat(frames, ignore_index=True)
