"""
Build data/sample_spectra.csv from the FTIR-Plastics spectrum PET100.

Source: FTIR-Plastics dataset, C4 set, spectrum PET100 (ATR-FTIR of a PET
bottle/container; Jasco FT/IR with ATR PRO 4x, 4 cm⁻¹ resolution, % transmittance).

    Villegas-Camacho, O., Alejo-Eleuterio, R., Francisco-Valencia, I.,
    Granda-Gutiérrez, E., Martínez-Gallegos, S. & Illescas, J. (2024).
    FTIR-Plastics: a Fourier Transform Infrared Spectroscopy dataset for the six
    most prevalent industrial plastic polymers. Zenodo.
    https://doi.org/10.5281/zenodo.10736650

    Paper: Data in Brief 55, 110612 (2024). https://doi.org/10.1016/j.dib.2024.110612

License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

Changes made:
  - dropped the 15-line metadata header;
  - cropped to 400–4000 cm⁻¹;
  - converted % transmittance to absorbance, A = -log10(T/100). Some points
    read above 100 %T (ATR baseline offset); T is clipped at 100, so they get A = 0;
  - sorted by descending wavenumber and rounded (4 decimals wavenumber,
    6 decimals absorbance).

Run from the project root, passing the raw PET100.csv from the dataset zip
(FTIR-PLASTIC-c4/PET_c4.zip):

    python scripts/make_sample_spectrum.py path/to/PET100.csv
"""

import sys
from pathlib import Path

import pandas as pd

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import convert_to_absorbance

N_HEADER_LINES = 15
OUTPUT_PATH = "data/sample_spectra.csv"


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/make_sample_spectrum.py path/to/PET100.csv")

    raw = pd.read_csv(sys.argv[1], skiprows=N_HEADER_LINES, header=None,
                      names=["wavenumber", "transmittance"])
    raw = raw[raw["wavenumber"].between(400, 4000)]

    absorbance = convert_to_absorbance(raw["transmittance"].to_numpy())
    out = pd.DataFrame({
        "wavenumber": raw["wavenumber"].round(4),
        "intensity": absorbance.round(6) + 0.0,  # + 0.0 turns -0.0 into 0.0
    })
    out = out.sort_values("wavenumber", ascending=False)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {len(out)} points → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
