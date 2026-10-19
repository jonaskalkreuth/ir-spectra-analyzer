# Data

## `sample_spectra.csv`

The bundled sample is spectrum **PET100** from the FTIR-Plastics dataset.
It is a real ATR-FTIR spectrum of a PET bottle/container
(Jasco FT/IR with ATR PRO 4x, 4 cm⁻¹ resolution).

- **Dataset:** Villegas-Camacho, O., Alejo-Eleuterio, R., Francisco-Valencia, I.,
  Granda-Gutiérrez, E., Martínez-Gallegos, S. & Illescas, J. (2024).
  *FTIR-Plastics: a Fourier Transform Infrared Spectroscopy dataset for the six most
  prevalent industrial plastic polymers.* Zenodo.
  https://doi.org/10.5281/zenodo.10736650
- **Paper:** Villegas-Camacho, O. et al. (2024). *FTIR-Plastics: A Fourier Transform
  Infrared Spectroscopy dataset for the six most prevalent industrial plastic polymers.*
  Data in Brief 55, 110612. https://doi.org/10.1016/j.dib.2024.110612
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Changes made** (by `scripts/make_sample_spectrum.py`):

- dropped the metadata header;
- cropped to 400–4000 cm⁻¹;
- converted % transmittance to absorbance, A = −log10(T/100), with T clipped at 100
  (points above 100 %T become A = 0);
- sorted by descending wavenumber and rounded.

## OpenSpecy training data

The OpenSpecy reference library used to train the classifier is **not included**.
Download it separately (see the main README).
