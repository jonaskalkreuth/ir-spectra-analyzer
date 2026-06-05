# ir-spectra-analyzer

FTIR signal processing pipeline and polymer classifier.

Classifies polymer types (PE, PP, PS, PET, …) from FTIR spectra using a
preprocessing pipeline (ALS baseline correction, Savitzky-Golay smoothing,
normalization) followed by SVM and Random Forest classifiers trained on the
OpenSpecy spectral library.

## Status

Work in progress — data acquisition and model training in progress.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python scripts/run_analysis.py --input data/sample_spectra.csv --plot
```

## Project structure

```
src/
  preprocessing.py   # ALS baseline, Savitzky-Golay, normalization
  peak_detection.py  # prominence-based peak detection
  loader.py          # CSV spectrum loader
  functional_groups.py  # IR functional group assignment table
  polymer_rules.py   # polymer-class rule-table baseline (FTIR peaks)
  classifier.py      # SVM + Random Forest polymer classifier
  analyzer.py        # full pipeline (preprocessing → peaks → groups)
  config.py          # shared defaults
scripts/
  run_analysis.py    # CLI entry point
data/
  sample_spectra.csv  # example single spectrum
  openspecy/          # OpenSpecy bulk download (not committed — see README)
models/               # trained model files (not committed)
notebooks/            # demo notebook
tests/
```
