from score_dense_public_ruler import ruler_score


def test_ruler_qa_accepts_any_reference() -> None:
    assert ruler_score("qa_1", "The answer is Paris.", ["Paris", "City of Paris"]) == 1.0


def test_ruler_multivalue_uses_fractional_all_reference_metric() -> None:
    assert ruler_score("niah_multivalue", "11 and 33", ["11", "22", "33", "44"]) == 0.5
