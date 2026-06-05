import pytest
import pandas as pd
import tempfile
import os
from src.loader import load_spectrum, load_multiple


VALID_CSV = "wavenumber,intensity\n4000,0.05\n3000,0.8\n2000,0.4\n1000,0.9\n"
ALIAS_CSV = "WN,ABS\n4000,0.05\n3000,0.8\n"
MISSING_ROW_CSV = "wavenumber,intensity\n4000,0.05\n,0.8\n2000,0.4\n"
BAD_COLUMN_CSV = "freq_nm,signal\n400,0.1\n300,0.5\n"


def _write_tmp(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def test_load_valid():
    path = _write_tmp(VALID_CSV)
    df = load_spectrum(path)
    assert list(df.columns) == ["wavenumber", "intensity"]
    assert len(df) == 4
    os.unlink(path)


def test_column_aliases():
    path = _write_tmp(ALIAS_CSV)
    df = load_spectrum(path)
    assert "wavenumber" in df.columns
    assert "intensity" in df.columns
    os.unlink(path)


def test_missing_row_dropped():
    path = _write_tmp(MISSING_ROW_CSV)
    with pytest.warns(UserWarning):
        df = load_spectrum(path)
    assert len(df) == 2
    os.unlink(path)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_spectrum("does_not_exist.csv")


def test_bad_columns():
    path = _write_tmp(BAD_COLUMN_CSV)
    with pytest.raises(ValueError, match="wavenumber column"):
        load_spectrum(path)
    os.unlink(path)


def test_load_multiple():
    p1 = _write_tmp(VALID_CSV)
    p2 = _write_tmp(VALID_CSV)
    df = load_multiple([p1, p2])
    assert "source" in df.columns
    assert len(df) == 8
    os.unlink(p1)
    os.unlink(p2)


def test_sample_data_loads():
    df = load_spectrum("data/sample_spectra.csv")
    assert len(df) > 10
    assert df["wavenumber"].between(400, 4000).all()
