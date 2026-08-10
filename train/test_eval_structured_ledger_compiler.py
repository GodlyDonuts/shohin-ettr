import pytest

from eval_structured_ledger_compiler import score_completion, shuffled_sources


GOLD = (
    "<LEDGER_V1>\nR0|SUB|8|3|5\nR1|ADD|@R0|5|10\n"
    "COMMIT|@R1|10\n</LEDGER_V1>"
)


def test_score_exact_and_terminal_only_difference():
    exact = score_completion(GOLD, GOLD)
    assert all(exact.values())
    changed = GOLD.replace("|10\nCOMMIT", "|11\nCOMMIT").replace("@R1|10", "@R1|11")
    score = score_completion(changed, GOLD)
    assert score["syntax_valid"] is True
    assert score["record_count_exact"] is True
    assert score["operation_sequence_exact"] is True
    assert score["all_records_exact"] is False
    assert score["terminal_exact"] is False


def test_score_rejects_prose_wrapper():
    score = score_completion("Here is the result:\n" + GOLD, GOLD)
    assert score["syntax_valid"] is False
    assert score["canonical_exact"] is False


def test_source_shuffle_is_stratified_and_changes_target():
    rows = [
        {
            "identity_sha256": f"{index:064x}",
            "family": "chain_sum",
            "record_count": 2,
            "response": f"target-{index}",
        }
        for index in range(3)
    ]
    donors = shuffled_sources(rows)
    assert set(donors) == {row["identity_sha256"] for row in rows}
    assert all(donors[row["identity_sha256"]]["response"] != row["response"] for row in rows)


def test_source_shuffle_rejects_singleton_stratum():
    with pytest.raises(ValueError, match="one row"):
        shuffled_sources(
            [{"identity_sha256": "a" * 64, "family": "x", "record_count": 1, "response": "y"}]
        )
