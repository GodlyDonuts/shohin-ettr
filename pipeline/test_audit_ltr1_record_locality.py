from audit_ltr1_record_locality import edit_distance, lcs_length, records


def test_records_extract_complete_transactions_only() -> None:
    assert records("x << (3 * 2) = 6 >> y <<broken") == ["(3*2)=6"]


def test_lcs_preserves_record_order() -> None:
    assert lcs_length(["a", "b", "c"], ["b", "a", "c"]) == 2


def test_record_edit_distance_supports_replace_insert_delete() -> None:
    assert edit_distance(["a", "b"], ["a", "c"]) == 1
    assert edit_distance(["a"], ["a", "b"]) == 1
    assert edit_distance(["a", "b"], ["a"]) == 1
