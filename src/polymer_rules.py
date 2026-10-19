# Rule-table baseline for FTIR material classification.
# Keys match material_class labels in the OpenSpecy metadata exactly.
# Each rule is a (lo, hi) wavenumber range in cm⁻¹; a rule is satisfied if
# any detected peak falls inside the range.
# Scoring: the class with the most matched rules wins (see classify_by_rules).
# Ranges are intentionally wider than textbook peak centres to absorb instrument
# resolution differences and slight peak shifts across samples.

POLYMER_RULES: dict[str, list[tuple[int, int]]] = {

    "poly(ethylene)": [
        (2900, 2940),   # asymmetric CH2 stretch
        (2840, 2870),   # symmetric CH2 stretch
        (1450, 1480),   # CH2 scissoring
        (710,  730),    # CH2 rocking — most diagnostic for PE
    ],

    "poly(propylene)": [
        (2940, 2965),   # asymmetric CH3 stretch
        (2900, 2930),   # asymmetric CH2 stretch
        (2860, 2880),   # symmetric CH3 stretch
        (1370, 1380),   # symmetric CH3 deformation — distinguishes PP from PE
        (990,  1010),   # CH3 rocking
        (830,  850),    # CH2 rocking
    ],

    "polystyrenes (polyphenylethylenes, -methylstyrene)": [
        (3010, 3110),   # aromatic C-H stretch
        (1580, 1620),   # aromatic C=C ring stretch
        (1480, 1510),   # aromatic C=C ring stretch (second band)
        (750,  770),    # aromatic C-H out-of-plane (monosubstituted)
        (690,  710),    # aromatic C-H out-of-plane (monosubstituted)
    ],

    # Catch-all for polyesters, polyethers, epoxies, and terephthalates.
    # Rules are intentionally broad; expect lower precision than PE/PP/PS.
    "poly(esters/ethers/diglycidylethers/terephthalates)s": [
        (1700, 1750),   # ester C=O stretch — present in all polyesters
        (1080, 1300),   # C-O-C stretch — esters and ethers
    ],

    # Polyamides (nylon family): amide I + II bands are very diagnostic.
    "poly(acrylamide/amid)s": [
        (3180, 3380),   # N-H stretch (amide A/B)
        (1620, 1680),   # amide I — C=O stretch
        (1510, 1570),   # amide II — N-H bending
    ],

    # Polyhaloolefins: C-Cl (PVC) or C-F (PTFE, PVDF) — at least one will fire.
    "polyhaloolefins (vinylhalides)": [
        (590,  710),    # C-Cl stretch (PVC)
        (1130, 1220),   # C-F stretch (PTFE/PVDF)
        (1000, 1080),   # C-F stretch (PVDF second band)
    ],

    # Minerals: silicate Si-O and carbonate C-O are the most common signatures.
    "mineral": [
        (950,  1150),   # Si-O-Si stretch (silicates — quartz, feldspar, clay)
        (1380, 1500),   # CO3 asymmetric stretch (calcite, dolomite)
        (3200, 3600),   # broad O-H stretch (clay minerals, hydroxides)
    ],

    # Organic matter: deliberately weak rules — this is a heterogeneous catch-all.
    # Its broad ranges overlap the specific classes; ties resolve to "Unknown".
    "organic matter": [
        (2840, 2970),   # aliphatic C-H stretch region — present in most organics
        (1580, 1760),   # C=O or C=C stretch region — broad organic signature
    ],
}


def classify_by_rules(detected_wavenumbers: list) -> str:
    """
    Assign a polymer class based on detected peak wavenumbers.

    Returns the polymer name with the most rule matches, or 'Unknown' if
    no rules are satisfied.
    """
    scores: dict = {}
    for polymer, rules in POLYMER_RULES.items():
        matched = sum(
            any(lo <= wn <= hi for wn in detected_wavenumbers)
            for lo, hi in rules
        )
        if matched > 0:
            scores[polymer] = matched

    if not scores:
        return "Unknown"
    top_score = max(scores.values())
    winners = [cls for cls, s in scores.items() if s == top_score]
    if len(winners) > 1:
        return "Unknown"
    return winners[0]
