"""
Spot-check the saved SVM on external spectra from the FTIR-Plastics dataset.

The FTIR-Plastics spectra are real ATR-FTIR measurements from an independent
lab and are not in the OpenSpecy training library, so they test how the model
transfers to new instruments.

    Villegas-Camacho, O., Alejo-Eleuterio, R., Francisco-Valencia, I.,
    Granda-Gutiérrez, E., Martínez-Gallegos, S. & Illescas, J. (2024).
    FTIR-Plastics: a Fourier Transform Infrared Spectroscopy dataset for the six
    most prevalent industrial plastic polymers. Zenodo.
    https://doi.org/10.5281/zenodo.10736650

    Paper: Data in Brief 55, 110612 (2024). https://doi.org/10.1016/j.dib.2024.110612

License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

Expected layout: the dataset folder with every per-polymer zip extracted, so
the raw CSVs sit anywhere below it, e.g.

    FTIR-Plastics/
        FTIR-PLASTIC-c4/
            PET_c4/PET100.csv
            HDPE_c4/HDPE1.csv
            ...

Zips are not read; unzip them first (e.g. `unzip PET_c4.zip -d PET_c4`).
The class comes from the letters before the number in each file name
(PET100.csv → PET). Files whose class is not in LABEL_MAP are skipped.

Each spectrum is read like scripts/make_sample_spectrum.py does it: drop the
15-line metadata header, crop to 400–4000 cm⁻¹, convert % transmittance to
absorbance. It then goes through `src.classifier.predict`, the same path the CLI
uses (derivative conversion, resampling, preprocessing, saved SVM).

Selection: per class, the first N files after sorting by path relative to the
dataset folder. The README's spot check (18 spectra) is this selection with
`--per-class 3`.

Run from the project root (the model is read from models/svm_classifier.joblib):

    python scripts/external_spot_check.py path/to/FTIR-Plastics [--per-class 3]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import load_model, predict
from src.preprocessing import convert_to_absorbance

N_HEADER_LINES = 15

# FTIR-Plastics class → model label. All six dataset polymers have a model class.
LABEL_MAP = {
    "PET": "poly(esters/ethers/diglycidylethers/terephthalates)s",
    "HDPE": "poly(ethylene)",
    "LDPE": "poly(ethylene)",
    "PP": "poly(propylene)",
    "PS": "polystyrenes (polyphenylethylenes, -methylstyrene)",
    "PVC": "polyhaloolefins (vinylhalides)",
}


def read_spectrum(path):
    """Read one raw FTIR-Plastics CSV as (wavenumbers, absorbance)."""
    raw = pd.read_csv(path, skiprows=N_HEADER_LINES, header=None,
                      names=["wavenumber", "transmittance"])
    raw = raw[raw["wavenumber"].between(400, 4000)]
    return (raw["wavenumber"].to_numpy(),
            convert_to_absorbance(raw["transmittance"].to_numpy()))


def main():
    parser = argparse.ArgumentParser(description="Spot-check the saved SVM on FTIR-Plastics spectra.")
    parser.add_argument("folder", type=Path, help="FTIR-Plastics folder with the zips extracted")
    parser.add_argument("--per-class", type=int, default=3, help="spectra per class (default 3)")
    args = parser.parse_args()

    # Group CSVs by class, in sorted relative-path order so the selection is reproducible.
    by_class, skipped = {}, set()
    for path in sorted(args.folder.rglob("*.csv"), key=lambda p: str(p.relative_to(args.folder))):
        match = re.match(r"[A-Za-z]+", path.stem)
        name = match.group(0).upper() if match else path.stem
        if name in LABEL_MAP:
            by_class.setdefault(name, []).append(path)
        else:
            skipped.add(name)

    if not by_class:
        sys.exit(f"no FTIR-Plastics CSVs found under {args.folder}")
    if skipped:
        print(f"skipped classes the model cannot predict: {', '.join(sorted(skipped))}")

    bundle = load_model()  # load once, reuse for every spectrum
    for name in sorted(by_class):
        paths = by_class[name][:args.per_class]
        misses = []
        for path in paths:
            wavenumbers, absorbance = read_spectrum(path)
            predicted = predict(wavenumbers, absorbance, bundle=bundle)["predicted_class"]
            if predicted != LABEL_MAP[name]:
                misses.append(f"{path.name} → {predicted}")
        print(f"{name:<5} {len(paths) - len(misses)}/{len(paths)}")
        for miss in misses:
            print(f"      miss: {miss}")


if __name__ == "__main__":
    main()
