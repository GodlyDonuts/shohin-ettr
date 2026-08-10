from eval_ectr0_executor_revision import receipt_text, shuffled_donors


def test_receipt_excludes_assessor_fields() -> None:
    detail = {
        "prediction": "7",
        "transactions": 2,
        "state_reads": 1,
        "source_reads": 2,
        "literal_reads": 0,
        "correct": True,
        "gold": "7",
    }
    receipt = receipt_text(detail)
    assert "result=7" in receipt
    assert "correct" not in receipt
    assert "gold" not in receipt


def test_shuffled_receipts_preserve_strata_without_identity() -> None:
    rows = [
        {"identity_sha256": "a", "register_depth": 2},
        {"identity_sha256": "b", "register_depth": 2},
        {"identity_sha256": "c", "register_depth": 3},
        {"identity_sha256": "d", "register_depth": 3},
    ]
    details = {
        "a": {"prediction": "1"},
        "b": {"prediction": "2"},
        "c": {},
        "d": {},
    }
    donors = shuffled_donors(rows, details)
    assert donors == {"a": "b", "b": "a", "c": "d", "d": "c"}
