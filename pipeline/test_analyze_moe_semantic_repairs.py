from analyze_moe_semantic_repairs import normalize_label, serialization_only


def test_label_normalization_unwraps_latex() -> None:
    assert normalize_label(r"\text{B}") == "b"


def test_label_prefix_is_certified_serialization() -> None:
    row = {"prediction": "B: a detailed explanation", "completion": ""}
    assert serialization_only(row, r"\text{B}")[0]


def test_semantic_answer_change_is_not_serialization() -> None:
    row = {"prediction": "C", "completion": r"\boxed{C}"}
    assert not serialization_only(row, "F")[0]


def test_numeric_units_are_certified_serialization() -> None:
    row = {"prediction": r"36.79 \text{ mol/L}", "completion": ""}
    assert serialization_only(row, "36.79")[0]
