from src.polymer_rules import classify_by_rules


def test_matches_characteristic_class():
    # The four diagnostic poly(ethylene) bands: CH2 stretches (asym/sym),
    # scissoring, and the rocking doublet at ~720 — all four rules fire,
    # nothing else scores as high, so PE wins outright.
    peaks = [2920, 2850, 1465, 720]
    assert classify_by_rules(peaks) == "poly(ethylene)"


def test_distinguishes_a_second_class():
    # Aromatic ring fingerprint (C-H stretch, two C=C ring bands, the
    # monosubstituted out-of-plane doublet) — only polystyrene's rules cover
    # this combination, confirming the table separates classes, not just
    # "finds peaks somewhere".
    peaks = [3050, 1600, 1490, 760, 700]
    assert classify_by_rules(peaks) == "polystyrenes (polyphenylethylenes, -methylstyrene)"


def test_no_peaks_is_unknown():
    # Empty peak list → no rule can match → the explicit "Unknown" guard.
    assert classify_by_rules([]) == "Unknown"


def test_tie_resolves_to_unknown():
    # A single peak at 2850 falls in BOTH poly(ethylene)'s sym-CH2 range
    # (2840–2870) and organic matter's aliphatic-CH range (2840–2970): each
    # class scores exactly 1, so the tie-breaker refuses to guess.
    assert classify_by_rules([2850]) == "Unknown"
