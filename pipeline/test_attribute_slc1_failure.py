from attribute_slc1_failure import arithmetic_consistent
from materialize_structured_ledger_sft import parse_ledger


def test_arithmetic_consistency_accepts_exact_chain():
    parsed = parse_ledger(
        "<LEDGER_V1>\nR0|SUB|8|3|5\nR1|ADD|@R0|5|10\n"
        "COMMIT|@R1|10\n</LEDGER_V1>"
    )
    assert arithmetic_consistent(parsed) is True


def test_arithmetic_consistency_rejects_invented_result():
    parsed = parse_ledger(
        "<LEDGER_V1>\nR0|SUB|8|3|6\nR1|ADD|@R0|5|11\n"
        "COMMIT|@R1|11\n</LEDGER_V1>"
    )
    assert arithmetic_consistent(parsed) is False
