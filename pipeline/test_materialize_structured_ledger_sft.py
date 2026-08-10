import json

import pytest

from materialize_structured_ledger_sft import (
    LedgerMaterializationError,
    encode_ledger,
    materialize,
    parse_ledger,
)


def ledger_row(identity: str, split: str) -> dict:
    return {
        "identity_sha256": identity * 64,
        "source_question_sha256": "a" * 64,
        "split": split,
        "family": "chain_sum",
        "question": "Compute 8 - 3 + 5.",
        "records": [
            {
                "address": 0,
                "operation": "SUB",
                "operands": [
                    {"numerator": 8, "denominator": 1},
                    {"numerator": 3, "denominator": 1},
                ],
                "result": {"numerator": 5, "denominator": 1},
                "dependencies": [],
            },
            {
                "address": 1,
                "operation": "ADD",
                "operands": [
                    {"numerator": 5, "denominator": 1},
                    {"numerator": 5, "denominator": 1},
                ],
                "result": {"numerator": 10, "denominator": 1},
                "dependencies": [{"operand_role": "left", "record_index": 0}],
            },
        ],
        "terminal_value": {"numerator": 10, "denominator": 1},
    }


def test_encode_and_parse_compact_ledger():
    row = ledger_row("b", "train")
    encoded = encode_ledger(row["records"], row["terminal_value"])
    assert encoded == (
        "<LEDGER_V1>\n"
        "R0|SUB|8|3|5\n"
        "R1|ADD|@R0|5|10\n"
        "COMMIT|@R1|10\n"
        "</LEDGER_V1>"
    )
    parsed = parse_ledger(encoded)
    assert parsed["records"][1]["left"] == "@R0"
    assert parsed["commit"] == {"address": "1", "value": "10"}


def test_encode_rejects_forward_dependency():
    row = ledger_row("b", "train")
    row["records"][0]["dependencies"] = [
        {"operand_role": "left", "record_index": 1}
    ]
    with pytest.raises(LedgerMaterializationError, match="not causal"):
        encode_ledger(row["records"], row["terminal_value"])


def test_materialize_hash_binds_and_preserves_splits(tmp_path):
    train_source = tmp_path / "train.jsonl"
    development_source = tmp_path / "development.jsonl"
    train_source.write_text(json.dumps(ledger_row("b", "train")) + "\n")
    development_source.write_text(json.dumps(ledger_row("c", "development")) + "\n")
    report = materialize(train_source, development_source, tmp_path / "output")
    assert report["canonical_round_trip_verified"] is True
    assert report["holdout_used"] is False
    assert report["counters"]["train"]["records"] == 2
    train_row = json.loads((tmp_path / "output" / "train.jsonl").read_text())
    assert train_row["split"] == "train"
    assert train_row["response"].startswith("<LEDGER_V1>\nR0|SUB")


def test_parse_rejects_commit_value_drift():
    with pytest.raises(LedgerMaterializationError, match="commit value differs"):
        parse_ledger(
            "<LEDGER_V1>\nR0|ADD|2|3|5\nCOMMIT|@R0|6\n</LEDGER_V1>"
        )
