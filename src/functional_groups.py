"""
IR functional group assignment based on characteristic absorption bands.

Each entry in FUNCTIONAL_GROUPS contains:
  - range: (low, high) wavenumber in cm⁻¹
  - intensity: typical band intensity ('strong', 'medium', 'weak', 'variable')
  - shape: typical band shape ('broad', 'sharp', 'variable')
  - description: short chemist-facing note on the vibration and compound context

Sources: Silverstein & Webster "Spectrometric Identification of Organic Compounds",
         Pavia et al. "Introduction to Spectroscopy".
"""

from dataclasses import dataclass
from collections import Counter
from typing import Optional


@dataclass(frozen=True)
class FunctionalGroup:
    name: str
    range: tuple
    intensity: str      # 'strong' | 'medium' | 'weak' | 'variable'
    shape: str          # 'broad' | 'sharp' | 'variable'
    description: str
    compound_class: str


FUNCTIONAL_GROUPS: list = [
    # ── O–H ────────────────────────────────────────────────────────────────
    FunctionalGroup(
        name="O-H (alcohol, free)",
        range=(3580, 3650),
        intensity="strong",
        shape="sharp",
        description="Free O-H stretch; dilute solution or gas phase alcohol.",
        compound_class="Alcohol / Phenol",
    ),
    FunctionalGroup(
        name="O-H (alcohol, H-bonded)",
        range=(3200, 3550),
        intensity="strong",
        shape="broad",
        description="Hydrogen-bonded O-H stretch; typical of neat alcohols and phenols.",
        compound_class="Alcohol / Phenol",
    ),
    FunctionalGroup(
        name="O-H (carboxylic acid)",
        range=(2500, 3300),
        intensity="strong",
        shape="broad",
        description="Very broad O-H stretch; diagnostic for carboxylic acids.",
        compound_class="Carboxylic Acid",
    ),
    # ── N–H ────────────────────────────────────────────────────────────────
    FunctionalGroup(
        name="N-H (primary amine)",
        range=(3300, 3500),
        intensity="medium",
        shape="sharp",
        description="Two N-H stretches (symmetric + asymmetric) for primary amines (-NH2).",
        compound_class="Amine",
    ),
    FunctionalGroup(
        name="N-H (secondary amine / amide)",
        range=(3300, 3400),
        intensity="medium",
        shape="sharp",
        description="Single N-H stretch for secondary amines and amides.",
        compound_class="Amine / Amide",
    ),
    FunctionalGroup(
        name="N-H (amide, H-bonded)",
        range=(3100, 3500),
        intensity="medium",
        shape="broad",
        description="Broadened N-H region in solid-state or H-bonded amides.",
        compound_class="Amide",
    ),
    # ── C–H ────────────────────────────────────────────────────────────────
    FunctionalGroup(
        name="=C-H (terminal alkyne)",
        range=(3280, 3340),
        intensity="strong",
        shape="sharp",
        description="Terminal alkyne C-H stretch; very sharp and strong.",
        compound_class="Alkyne",
    ),
    FunctionalGroup(
        name="=C-H (aromatic / vinyl)",
        range=(3000, 3100),
        intensity="medium",
        shape="sharp",
        description="Aromatic and vinyl C-H stretches above 3000 cm-1.",
        compound_class="Aromatic / Alkene",
    ),
    FunctionalGroup(
        name="C-H (alkyl, asymmetric stretch)",
        range=(2950, 2970),
        intensity="strong",
        shape="sharp",
        description="Asymmetric CH3 / CH2 stretch; ubiquitous in organic molecules.",
        compound_class="Alkane / Alkyl",
    ),
    FunctionalGroup(
        name="C-H (alkyl, symmetric stretch)",
        range=(2840, 2870),
        intensity="strong",
        shape="sharp",
        description="Symmetric CH3 / CH2 stretch.",
        compound_class="Alkane / Alkyl",
    ),
    FunctionalGroup(
        name="C-H (aldehyde)",
        range=(2700, 2850),
        intensity="medium",
        shape="sharp",
        description="Two weak C-H stretches (~2820 and ~2720 cm-1) diagnostic for aldehydes.",
        compound_class="Aldehyde",
    ),
    # ── Triple bonds ────────────────────────────────────────────────────────
    FunctionalGroup(
        name="C=N (nitrile)",
        range=(2200, 2260),
        intensity="strong",
        shape="sharp",
        description="C=N stretch; strong and sharp, diagnostic for nitriles.",
        compound_class="Nitrile",
    ),
    FunctionalGroup(
        name="C=C (internal alkyne)",
        range=(2100, 2260),
        intensity="variable",
        shape="sharp",
        description="C=C stretch; absent for symmetric internal alkynes (IR-inactive).",
        compound_class="Alkyne",
    ),
    FunctionalGroup(
        name="N=C=O (isocyanate)",
        range=(2250, 2275),
        intensity="strong",
        shape="broad",
        description="Asymmetric N=C=O stretch; very strong and broad.",
        compound_class="Isocyanate",
    ),
    # ── Carbonyl C=O ────────────────────────────────────────────────────────
    FunctionalGroup(
        name="C=O (acid anhydride, high freq)",
        range=(1800, 1850),
        intensity="strong",
        shape="sharp",
        description="Higher-frequency carbonyl of anhydrides (two C=O bands expected).",
        compound_class="Anhydride",
    ),
    FunctionalGroup(
        name="C=O (acid chloride)",
        range=(1770, 1815),
        intensity="strong",
        shape="sharp",
        description="Acid chloride carbonyl; shifted higher by inductive effect of Cl.",
        compound_class="Acyl Halide",
    ),
    FunctionalGroup(
        name="C=O (acid anhydride, low freq)",
        range=(1750, 1790),
        intensity="strong",
        shape="sharp",
        description="Lower-frequency (symmetric) carbonyl of anhydrides.",
        compound_class="Anhydride",
    ),
    FunctionalGroup(
        name="C=O (ester)",
        range=(1735, 1750),
        intensity="strong",
        shape="sharp",
        description="Ester carbonyl; accompanied by strong C-O-C stretch ~1200 cm-1.",
        compound_class="Ester",
    ),
    FunctionalGroup(
        name="C=O (aldehyde)",
        range=(1720, 1740),
        intensity="strong",
        shape="sharp",
        description="Aldehyde carbonyl; accompanied by aldehyde C-H doublet ~2720/2820 cm-1.",
        compound_class="Aldehyde",
    ),
    FunctionalGroup(
        name="C=O (ketone)",
        range=(1705, 1725),
        intensity="strong",
        shape="sharp",
        description="Ketone carbonyl; shifted lower than aldehyde/ester.",
        compound_class="Ketone",
    ),
    FunctionalGroup(
        name="C=O (carboxylic acid)",
        range=(1700, 1725),
        intensity="strong",
        shape="sharp",
        description="Acid carbonyl; combined with broad O-H stretch 2500-3300 cm-1.",
        compound_class="Carboxylic Acid",
    ),
    FunctionalGroup(
        name="C=O (conjugated enone / enal)",
        range=(1665, 1685),
        intensity="strong",
        shape="sharp",
        description="alpha,beta-unsaturated carbonyl; shifted lower by conjugation.",
        compound_class="Enone / Enal",
    ),
    FunctionalGroup(
        name="C=O (amide I)",
        range=(1630, 1690),
        intensity="strong",
        shape="sharp",
        description="Amide I band (C=O stretch); lowered by resonance with N lone pair.",
        compound_class="Amide",
    ),
    # ── C=C ─────────────────────────────────────────────────────────────────
    FunctionalGroup(
        name="C=C (alkene)",
        range=(1620, 1680),
        intensity="variable",
        shape="sharp",
        description="Alkene C=C stretch; weak for symmetric alkenes (may be IR-inactive).",
        compound_class="Alkene",
    ),
    FunctionalGroup(
        name="C=C (aromatic ring)",
        range=(1450, 1600),
        intensity="variable",
        shape="sharp",
        description="Aromatic ring C=C stretches; often two bands at ~1500 and ~1600 cm-1.",
        compound_class="Aromatic",
    ),
    # ── N–H bending / amide II ──────────────────────────────────────────────
    FunctionalGroup(
        name="N-H bend (primary amine)",
        range=(1550, 1650),
        intensity="medium",
        shape="sharp",
        description="NH2 scissoring; confirms primary amine when paired with N-H stretch.",
        compound_class="Amine",
    ),
    FunctionalGroup(
        name="N-H bend / C-N stretch (amide II)",
        range=(1510, 1560),
        intensity="strong",
        shape="sharp",
        description="Amide II band (N-H bend + C-N stretch coupling); secondary amides.",
        compound_class="Amide",
    ),
    # ── NO₂ ─────────────────────────────────────────────────────────────────
    FunctionalGroup(
        name="NO2 (asymmetric stretch)",
        range=(1500, 1570),
        intensity="strong",
        shape="sharp",
        description="Asymmetric N=O stretch of nitro group; very strong.",
        compound_class="Nitro",
    ),
    FunctionalGroup(
        name="NO2 (symmetric stretch)",
        range=(1340, 1390),
        intensity="strong",
        shape="sharp",
        description="Symmetric N=O stretch of nitro group.",
        compound_class="Nitro",
    ),
    # ── S=O ─────────────────────────────────────────────────────────────────
    FunctionalGroup(
        name="S=O (sulfone, asymmetric)",
        range=(1290, 1350),
        intensity="strong",
        shape="sharp",
        description="Asymmetric SO2 stretch of sulfones.",
        compound_class="Sulfone",
    ),
    FunctionalGroup(
        name="S=O (sulfone, symmetric)",
        range=(1120, 1160),
        intensity="strong",
        shape="sharp",
        description="Symmetric SO2 stretch of sulfones.",
        compound_class="Sulfone",
    ),
    FunctionalGroup(
        name="S=O (sulfoxide)",
        range=(1030, 1070),
        intensity="strong",
        shape="sharp",
        description="S=O stretch of sulfoxides.",
        compound_class="Sulfoxide",
    ),
    # ── C–O single bond ─────────────────────────────────────────────────────
    FunctionalGroup(
        name="C-O-C (ester)",
        range=(1150, 1250),
        intensity="strong",
        shape="sharp",
        description="C-O-C asymmetric stretch of esters; often the strongest fingerprint band.",
        compound_class="Ester",
    ),
    FunctionalGroup(
        name="C-O (ether)",
        range=(1070, 1150),
        intensity="strong",
        shape="sharp",
        description="Asymmetric C-O-C stretch of ethers.",
        compound_class="Ether",
    ),
    FunctionalGroup(
        name="C-O (tertiary alcohol)",
        range=(1150, 1200),
        intensity="strong",
        shape="sharp",
        description="C-O stretch of tertiary alcohols.",
        compound_class="Alcohol",
    ),
    FunctionalGroup(
        name="C-O (secondary alcohol)",
        range=(1075, 1150),
        intensity="strong",
        shape="sharp",
        description="C-O stretch of secondary alcohols.",
        compound_class="Alcohol",
    ),
    FunctionalGroup(
        name="C-O (primary alcohol)",
        range=(1020, 1075),
        intensity="strong",
        shape="sharp",
        description="C-O stretch of primary alcohols.",
        compound_class="Alcohol",
    ),
    # ── Aromatic out-of-plane C–H ────────────────────────────────────────────
    FunctionalGroup(
        name="=C-H oop (monosubstituted aromatic)",
        range=(730, 770),
        intensity="strong",
        shape="sharp",
        description="Aromatic C-H out-of-plane bend; monosubstituted ring (5 adjacent H).",
        compound_class="Aromatic",
    ),
    FunctionalGroup(
        name="=C-H oop (para-disubstituted aromatic)",
        range=(800, 860),
        intensity="strong",
        shape="sharp",
        description="Aromatic C-H out-of-plane bend; para-disubstituted ring (2 adjacent H).",
        compound_class="Aromatic",
    ),
    # ── C–Halogen ────────────────────────────────────────────────────────────
    FunctionalGroup(
        name="C-Cl (alkyl chloride)",
        range=(600, 800),
        intensity="strong",
        shape="sharp",
        description="C-Cl stretch; useful for identifying chloroalkanes.",
        compound_class="Alkyl Halide",
    ),
    FunctionalGroup(
        name="C-F (alkyl fluoride)",
        range=(1000, 1400),
        intensity="strong",
        shape="sharp",
        description="C-F stretch; very strong, broad region.",
        compound_class="Alkyl Halide",
    ),
    # ── Other ────────────────────────────────────────────────────────────────
    FunctionalGroup(
        name="P=O (phosphate / phosphonate)",
        range=(1150, 1300),
        intensity="strong",
        shape="sharp",
        description="P=O stretch; strong absorption in organophosphates.",
        compound_class="Organophosphate",
    ),
]

_INTENSITY_RANK = {"strong": 3, "medium": 2, "weak": 1, "variable": 1}


def assign_groups(
    wavenumbers,
    min_intensity: Optional[str] = None,
) -> list:
    """
    Assign functional groups to a list of peak wavenumbers.

    Returns a list of dicts (one per match), sorted by wavenumber descending.
    Each dict has keys: wavenumber, group, compound_class, intensity, shape,
    description.

    Parameters
    ----------
    wavenumbers:
        Peak positions in cm⁻¹ (list or array).
    min_intensity:
        If given ('strong', 'medium', 'weak'), only return matches at or above
        this intensity level. Useful to filter out noise hits.
    """
    min_rank = _INTENSITY_RANK.get(min_intensity or "weak", 1)
    matches = []

    for wn in wavenumbers:
        for fg in FUNCTIONAL_GROUPS:
            lo, hi = fg.range
            if lo <= wn <= hi and _INTENSITY_RANK.get(fg.intensity, 1) >= min_rank:
                matches.append(
                    {
                        "wavenumber": float(wn),
                        "group": fg.name,
                        "compound_class": fg.compound_class,
                        "intensity": fg.intensity,
                        "shape": fg.shape,
                        "description": fg.description,
                    }
                )

    matches.sort(key=lambda m: m["wavenumber"], reverse=True)
    return matches


def summarize_compound_classes(matches: list) -> list:
    """Return unique compound classes found, ordered by match frequency."""
    counts = Counter(m["compound_class"] for m in matches)
    return [cls for cls, _ in counts.most_common()]
