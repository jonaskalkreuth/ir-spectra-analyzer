import pytest
from src.analyzer import analyze


def test_analyze_returns_dict():
    result = analyze("data/sample_spectra.csv")
    assert isinstance(result, dict)


def test_analyze_required_keys():
    result = analyze("data/sample_spectra.csv")
    for key in ("source", "num_peaks", "peaks", "groups", "compound_classes"):
        assert key in result, f"Missing key: {key}"


def test_analyze_peak_count_consistent():
    result = analyze("data/sample_spectra.csv")
    assert result["num_peaks"] == len(result["peaks"])


def test_analyze_peak_dict_structure():
    result = analyze("data/sample_spectra.csv")
    for p in result["peaks"]:
        assert "wavenumber" in p
        assert "intensity" in p
        assert "prominence" in p
        assert "fwhm_cm" in p
        assert "region" in p
        assert 400 <= p["wavenumber"] <= 4000


def test_analyze_groups_dict_structure():
    result = analyze("data/sample_spectra.csv")
    for g in result["groups"]:
        assert "group" in g
        assert "wavenumber" in g
        assert "compound_class" in g


def test_analyze_missing_file():
    with pytest.raises(FileNotFoundError):
        analyze("no_such_file.csv")


def test_analyze_no_baseline():
    result = analyze("data/sample_spectra.csv", correct_baseline=False)
    assert isinstance(result, dict)
    assert result["num_peaks"] >= 0


def test_analyze_transmittance_mode():
    # % Transmittance between 0 and 100 — just check it doesn't crash
    import tempfile, os
    content = "wavenumber,transmittance\n4000,99\n3400,30\n2900,40\n1720,5\n1000,20\n"
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    result = analyze(path, input_is_transmittance=True, correct_baseline=False)
    assert isinstance(result, dict)
    os.unlink(path)
