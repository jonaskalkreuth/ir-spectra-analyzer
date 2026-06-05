import pytest
from src.functional_groups import assign_groups, summarize_compound_classes, FUNCTIONAL_GROUPS


def test_carbonyl_matched():
    # 1715 cm-1 is firmly in ketone / carboxylic acid C=O range
    matches = assign_groups([1715.0])
    groups = [m["group"] for m in matches]
    assert any("C=O" in g for g in groups)


def test_oh_stretch_matched():
    matches = assign_groups([3400.0])
    classes = [m["compound_class"] for m in matches]
    assert any("Alcohol" in c or "Amine" in c for c in classes)


def test_no_match_outside_all_ranges():
    # 200 cm-1 is below all defined ranges
    matches = assign_groups([200.0])
    assert matches == []


def test_returns_sorted_descending():
    matches = assign_groups([1000.0, 3000.0, 1700.0])
    wns = [m["wavenumber"] for m in matches]
    assert wns == sorted(wns, reverse=True)


def test_intensity_filter_strong_only():
    # 1715 cm-1 should have 'strong' C=O matches
    all_matches = assign_groups([1715.0])
    strong_matches = assign_groups([1715.0], min_intensity="strong")
    assert len(strong_matches) <= len(all_matches)
    for m in strong_matches:
        assert m["intensity"] == "strong"


def test_match_dict_keys():
    matches = assign_groups([1715.0])
    assert len(matches) > 0
    expected_keys = {"wavenumber", "group", "compound_class", "intensity", "shape", "description"}
    assert expected_keys <= set(matches[0].keys())


def test_summarize_compound_classes():
    matches = assign_groups([3400.0, 3300.0, 1715.0])
    classes = summarize_compound_classes(matches)
    assert isinstance(classes, list)
    assert len(classes) > 0


def test_all_groups_have_required_fields():
    for fg in FUNCTIONAL_GROUPS:
        assert fg.name
        assert len(fg.range) == 2
        assert fg.range[0] < fg.range[1]
        assert fg.intensity in ("strong", "medium", "weak", "variable")
        assert fg.shape in ("broad", "sharp", "variable")
        assert fg.description
        assert fg.compound_class
