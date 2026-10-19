import pytest

from src import data_acquisition
from src.data_acquisition import download_openspecy


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    # Fake the download (no network): wrong bytes must raise, name both hashes,
    # and leave neither the destination nor the .part file behind.
    def fake_urlretrieve(url, path):
        path.write_bytes(b"not the library")

    monkeypatch.setattr(data_acquisition.urllib.request, "urlretrieve", fake_urlretrieve)
    dest = tmp_path / "derivative.rds"

    with pytest.raises(RuntimeError, match="Checksum mismatch") as exc:
        download_openspecy(dest)
    assert data_acquisition._RDS_SHA256 in str(exc.value)
    assert "Got:" in str(exc.value)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()
