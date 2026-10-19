# ir-spectra-analyzer

[![CI](https://github.com/jonaskalkreuth/ir-spectra-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/jonaskalkreuth/ir-spectra-analyzer/actions/workflows/ci.yml)

**Classify the material behind an FTIR spectrum — polymer, mineral, or organic matter — and read the chemistry the model used to decide.**

**[Demo notebook →](notebooks/demo.ipynb)**

A reproducible FTIR pipeline that takes a raw infrared spectrum, runs the
classical signal-processing chain a spectroscopist would (baseline correction,
smoothing, peak detection, functional-group assignment), and then predicts the
**material class** with a machine-learning model trained on the open
[OpenSpecy](https://www.openspecy.org/) reference library. It ships an
interpretable rule-table baseline, a correlation library-matching baseline (the
standard method instruments actually use), an SVM, and a Random Forest, compared
head-to-head with leakage-aware evaluation — the project is as much about
*careful method validation* as about the final score.

The 8 classes span polymers, minerals, and organic matter, so the task mirrors
a real industrial use case: **microplastic screening — telling polymers apart
*from* the minerals and organic debris they're found alongside.**

## Results

Four classifiers on the **same** held-out 20% test set (4,113 spectra never
seen during training or tuning), scored by **F1-macro** over the 8 classes
(every class weighted equally, so rare classes can't hide behind the common ones):

| Model | F1-macro (all 4,113 test spectra) | F1-macro (1,199 with no training neighbour at r ≥ 0.99) |
|---|---:|---:|
| Correlation library matching (1-NN) | **0.987** | **0.968** |
| SVM (RBF kernel) | 0.976 | 0.949 |
| Random Forest | 0.957 | — |
| Rule-table baseline | 0.109 | — |

**Read the headline with its caveat.** The library has many near-twins: 71% of
test spectra have a training spectrum at Pearson r ≥ 0.99, and 31% at
r ≥ 0.999. The second column drops those and scores only the remaining 1,199
test spectra. There the SVM falls from 0.976 to 0.949 (see
[Methodology & caveats](#methodology--caveats)).

![SVM confusion matrix](reports/confusion_svm.png)

The headline isn't "ML wins".
**Correlation library matching** — the standard way an instrument identifies a
spectrum, by Pearson-correlating it against every reference and taking the
closest match — tops the table at 0.987. On a *curated reference library* this
is expected: most test spectra have a near-twin among the same-class
references, so a nearest-neighbour lookup is hard to beat. The learned models
land within ~1 F1-point, and the SVM **keeps ~3.1k support vectors (19% of the
library) instead of all 16k** at query time. The SVM's confusion matrix is
near-diagonal — every class 0.88–1.00, the only notable leak being
`poly(acrylamide/amid)s` (recall 0.88) occasionally read as mineral — and the
Random Forest trails slightly but earns its place by being **interpretable**
(see below). The rule baseline scores poorly for a known reason — see
[Methodology & caveats](#methodology--caveats).

## Methodology & caveats

The headline number is easy to inflate by accident. The choices below are the
guardrails against that — and the part a reviewer should read most closely.

- **Leakage-aware evaluation.** A stratified 20% slice is locked away *before*
  any tuning. Hyperparameters are chosen by cross-validation on the remaining
  80% only; the final F1-macro is measured **once** on the untouched 20%. No
  parameter ever "sees" the test set.
- **Per-fold scaling.** The SVM's `StandardScaler` is fit *inside* each
  cross-validation fold (via a scikit-learn `Pipeline`), so a fold's own
  mean/variance never leaks into its training. The Random Forest uses **no
  scaler** at all — tree splits are scale-invariant, so standardizing would be
  a misleading no-op. Preprocessing is model-dependent, and the code says so.
- **Near-twins inflate the headline score.** OpenSpecy is a *curated
  reference* library, and exact duplicates are rare (1 / 21,542, dropped).
  Near-duplicates are not rare. On the seed-42 split, 31% of test spectra have a
  training neighbour at Pearson r ≥ 0.999, 71% at r ≥ 0.99, and 86% at
  r ≥ 0.95. A plain stratified split therefore rewards memorising the library.
  On the 1,199 test spectra with no neighbour at r ≥ 0.99, the SVM scores 0.949
  and correlation matching 0.968. On the 568 with no neighbour at r ≥ 0.95, they
  score 0.873 and 0.920, but that subset is noisy (11–65 spectra in each
  non-mineral class). `scripts/measure_near_duplicates.py` reproduces these
  numbers. A group-aware split would remove the effect; it is not done here.
- **Class imbalance is weighted, not dropped.** The rare tail is cut **once**
  (an 8-class, ≥300-spectra cutoff → 20,563 spectra; the smallest class kept is
  `polyhaloolefins` with 301), then every surviving sample of every class is
  kept and `class_weight='balanced'` handles the remaining imbalance. No
  down-sampling of the common classes.
- **Two baselines — a floor and a ceiling.** The interpretable rule table is
  the *floor*; **correlation library matching** is the *ceiling* — the method
  instruments actually use (1-NN by Pearson correlation against the training
  references). It scores **0.987**, edging out both ML models. On curated
  reference data the standard method is already excellent, so the learned
  models' real contribution is *matching* it with a smaller model, not beating
  it on accuracy. Where a learned boundary might pull ahead — weathered, mixed,
  off-library field spectra — is **out of scope here and not claimed**, only
  flagged.
- **Why the rule baseline scores 0.109.** The rules list characteristic
  absorbance band centres per class. The OpenSpecy library stores spectra as
  absolute first derivatives, and in that form each band centre becomes a
  *dip*, with a peak on either flank. So the rules look for peaks where the
  training data has minima, and each real band yields two detected peaks
  (~18 per spectrum). The rule counts then tie constantly, ties resolve to
  `"Unknown"`, and the baseline predicts `"Unknown"` for 60% of test spectra.
  The rules were not re-tuned to the derivative form; the score shows the
  representation mismatch, not the limit of a rule-based approach.
- **The Random Forest learned the chemistry.** Its feature importances peak on
  the diagnostic IR bands — C–H stretch, the C=O carbonyl region (~1745 cm⁻¹,
  the single most important feature), C–O / C–N, and O–H (~3650 cm⁻¹) —
  recovered with no chemical priors, purely from the labelled spectra.
- **Known weakness: polystyrene from an independent lab.** A spot check on 12
  spectra from the external FTIR-Plastics dataset (real ATR-FTIR, not in the
  training library) got PET 3/3, HDPE 3/3 and PP 2/3 right, but PS 0/3.
  The PS spectra were read as polyethylene or polypropylene. The cause is not
  yet investigated, and a full external test is still to do.

![RF feature importances on the wavenumber axis](reports/rf_importances.png)
![RF confusion matrix](reports/confusion_rf.png)

The 8 classes are: `poly(ethylene)`, `poly(propylene)`, `polystyrenes`,
`poly(esters/ethers/diglycidylethers/terephthalates)s`,
`poly(acrylamide/amid)s`, `polyhaloolefins`, `mineral`, and `organic matter`.
The last three (and the polyester family) are deliberately broad catch-all
buckets — a single `mineral` label covers many distinct materials — which makes
the classification problem realistically hard rather than a clean toy set.

## Install

Requires **Python 3.11**. Install the pinned versions used for every reported
number:

```bash
pip install -r requirements-lock.txt
```

`requirements.txt` lists the same packages with loose version bounds. Use it
if you can't run Python 3.11. Results may then differ slightly, and the model
may need retraining under the installed scikit-learn.

Pure Python for everyday use. **Reproducing the dataset from scratch**
additionally needs R + the [OpenSpecy R package](https://github.com/wincowgerDEV/OpenSpecy-package)
(used once to export the reference library — see
[Reproduce from scratch](#reproduce-from-scratch)).

## Usage

Analyze a single spectrum (CSV with `wavenumber,intensity` columns, absorbance):

```bash
python scripts/run_analysis.py --input data/sample_spectra.csv
```

The bundled sample is a real PET spectrum from an independent lab, not part of
the training library (see [Data & license](#data--license)). The CLI prints the
spectrum report — detected peaks, functional-group assignments, likely compound
classes — **and** the model's prediction (trimmed here):

```
  Peaks detected: 14

    Wavenumber   Intensity  Prominence  FWHM (cm-1)  Region
  ------------------------------------------------------
      1712.5      0.7611      0.7570         30.2  Double bond
      1239.0      0.6998      0.6969         53.4  Fingerprint
      1093.4      0.6416      0.6376         55.4  Fingerprint
       723.2      1.0000      0.9973         22.9  Fingerprint
    ...

  Possible functional-group assignments (window match only) (14 matches):

    1239.0 cm-1  C-O-C (ester)                             [strong, sharp]
           -> C-O-C asymmetric stretch of esters; often the strongest fingerprint band.
    1093.4 cm-1  C-O (ether)                               [strong, sharp]
           -> Asymmetric C-O-C stretch of ethers.
    ...

  Predicted material: poly(esters/ethers/diglycidylethers/terephthalates)s  (99.9% confidence)
  Runners-up: mineral 0.1%, polystyrenes (polyphenylethylenes, -methylstyrene) 0.0%
```

The runners-up are shown deliberately: a confident "99% polyester" and a
coin-flip "51% mineral / 47% PE" are very different signals, and a
chemometrician wants to see the model's uncertainty, not just its top guess.

**The model identifies the material; the functional-group report does not.**
The report matches each peak against fixed wavenumber windows, with no context
rules. On this PET spectrum it also lists C–F, C–Cl, nitro, sulfone and P=O
hits in the fingerprint region, which are false positives. It also reads PET's
conjugated ester C=O at 1712 cm⁻¹ as ketone or carboxylic acid.

**Input is absorbance; the model sees derivatives.** The OpenSpecy training
library holds absolute first-derivative spectra. Before classifying, the CLI
converts the input the same way OpenSpecy does: resample to 5 cm⁻¹,
Savitzky-Golay first derivative (window 11, polynomial 3), absolute value,
then min-max scaling to 0–1.

Useful flags:

| Flag | Effect |
|---|---|
| `--no-classify` | Skip ML classification; print the spectrum report only |
| `--output results.json` | Save the full report + prediction as JSON |
| `--plot` | Show the spectrum with annotated peaks |
| `--transmittance` | Treat input as %T and convert to absorbance (for the report and the model) |
| `--no-baseline` | Skip ALS baseline correction |

Classification is **on by default** and degrades gracefully: the trained model
(`models/*.joblib`) isn't committed, so on a fresh clone the CLI prints the
report plus a one-line hint pointing to
[Reproduce from scratch](#reproduce-from-scratch), rather than crashing.

## Reproduce from scratch

The OpenSpecy download and the derived `.npz` artifacts are gitignored (large,
regenerable). To rebuild everything:

```bash
# 1. Download the OpenSpecy reference library and export it to CSV.
#    download_openspecy() fetches derivative.rds from OSF (DOI 10.17605/OSF.IO/X7DPZ);
#    export_to_csv() shells out to Rscript (the .rds is an OpenSpecy R6 object that
#    Python can't unpickle), producing wavenumbers/spectra/metadata CSVs, FTIR-only.
python -c "from src.data_acquisition import download_openspecy, export_to_csv; \
           download_openspecy(); export_to_csv()"

# 2. Assemble the training set: measure duplicates, drop exact dupes,
#    filter to the 8 best-populated classes, save training_set.npz.
python -c "from src.classifier import assemble_training_set; assemble_training_set()"

# 3. Build the feature matrix: resample each spectrum to the fixed
#    600–4000 cm⁻¹ / 4 cm⁻¹ grid (851 features), preprocess, cache features.npz.
python -c "from src.classifier import build_features; build_features()"

# 4. Train + compare rule baseline vs correlation matching vs SVM vs RF;
#    write the figures to reports/.
python scripts/compare_models.py

# 5. Persist the winning SVM (refit with probability=True for confidences).
python scripts/persist_model.py

# Optional: measure the near-twin share and score the no-twin subsets.
python scripts/measure_near_duplicates.py
```

After step 5 the CLI in [Usage](#usage) will classify against the freshly
trained model.

## How it works

1. **Signal processing** (`preprocessing.py`, `peak_detection.py`) — ALS
   baseline correction, Savitzky-Golay smoothing, min-max normalization, then
   prominence-based peak detection.
2. **Feature engineering** (`classifier.py`) — each spectrum is resampled onto a
   fixed **600–4000 cm⁻¹ grid at 4 cm⁻¹ steps (851 features)** and fed to the ML
   models *as the full spectrum*. Training spectra are already in OpenSpecy's
   derivative form; at inference, `predict()` first converts absorbance input
   to that form. Peaks are **not** the ML feature — they feed the
   human-readable report and the rule baseline only.
3. **Two parallel "opinions"** — a classical, interpretable rule-table baseline
   (`polymer_rules.py`, characteristic peak ranges per class) and the trained
   ML models (`classifier.py`). The CLI shows both the spectroscopist's report
   and the model's prediction.

## Project structure

```
src/
  loader.py            # CSV spectrum loader
  preprocessing.py     # ALS baseline, Savitzky-Golay, normalization, OpenSpecy derivative
  peak_detection.py    # prominence-based peak detection
  functional_groups.py # IR functional-group assignment table (the report)
  polymer_rules.py     # rule-table baseline (characteristic peaks per class)
  data_acquisition.py  # OpenSpecy download + R-export-to-CSV + loader
  classifier.py        # feature engineering, SVM + RF, compare, persist, predict
  analyzer.py          # end-to-end report (preprocess → peaks → groups)
  utils.py             # report printing, JSON export, plotting
  config.py            # shared constants (grid, CV params, paths)
scripts/
  run_analysis.py              # CLI entry point (report + prediction)
  compare_models.py            # all four models → reports/ figures
  persist_model.py             # train + save the winning model bundle
  measure_near_duplicates.py   # near-twin share + no-twin subset scores
  make_sample_spectrum.py      # builds data/sample_spectra.csv from FTIR-Plastics
notebooks/
  demo.ipynb           # walkthrough: report, prediction, model comparison, held-out gallery
data/
  sample_spectra.csv   # demo spectrum (real PET, see data/README.md)
  openspecy/           # OpenSpecy download + derived .npz (gitignored)
models/                # trained model bundle (gitignored)
reports/               # committed comparison figures
tests/                 # pytest suite (59 tests)
requirements-lock.txt  # pinned versions (Python 3.11)
requirements.txt       # loose version bounds
LICENSE                # MIT (code only)
```

## Testing

```bash
python -m pytest
```

The suite (59 tests) covers the signal-processing half and the ML half,
including a synthetic-model fixture so the classifier tests run on a fresh
clone with no downloaded data. One integration test needs the trained model
and is skipped without it. CI runs the suite on every push and pull request.

## Data & license

**Code:** MIT — see [`LICENSE`](LICENSE). The license covers the code only.

**Training data (not included):** the OpenSpecy reference library, downloaded
from OSF (DOI [10.17605/OSF.IO/X7DPZ](https://doi.org/10.17605/OSF.IO/X7DPZ)).
FTIR spectra only; Raman filtered out for a single consistent modality. The
OSF project lists its license as "Other"; check its terms before reuse. If you
use the library, cite:

> Cowger, W. et al. (2025). Open Specy 1.0: Automated (Hyper)spectroscopy for
> Microplastics. *Analytical Chemistry* 97(32), 17345–17356.
> https://doi.org/10.1021/acs.analchem.5c00962

**Sample spectrum (included):** `data/sample_spectra.csv` is spectrum PET100
from the FTIR-Plastics dataset, CC BY 4.0, converted to absorbance and cropped.
[`data/README.md`](data/README.md) lists the changes.

> Villegas-Camacho, O. et al. (2024). FTIR-Plastics: a Fourier Transform
> Infrared Spectroscopy dataset for the six most prevalent industrial plastic
> polymers. Zenodo. https://doi.org/10.5281/zenodo.10736650 — described in
> *Data in Brief* 55, 110612. https://doi.org/10.1016/j.dib.2024.110612
