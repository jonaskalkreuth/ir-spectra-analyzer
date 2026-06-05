# Polymer rule-table baseline classifier.
# Each entry: polymer name → list of characteristic FTIR peak ranges (cm⁻¹).
# A spectrum "matches" a polymer if it has a detected peak inside each required range.
# TODO: populated in step B1b-1a-iii once OpenSpecy class distribution is known.

POLYMER_RULES: dict = {
    # Example structure — to be filled with real peak rules per polymer class.
    # "PE": [
    #     (2840, 2870),   # symmetric CH2 stretch
    #     (2910, 2930),   # asymmetric CH2 stretch
    #     (715, 730),     # CH2 rocking
    # ],
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
    return max(scores, key=lambda k: scores[k])
