import json

import pytest

from build_structured_ledger_edit_pairs import (
    LedgerEditPairError,
    apply_edit_script,
    build,
    fault_ledger,
    keep_script,
    replace_script,
)


GOLD = (
    "<LEDGER_V1>\nR0|SUB|8|3|5\nR1|ADD|@R0|5|10\n"
    "COMMIT|@R1|10\n</LEDGER_V1>"
)


def row(identity, split):
    return {
        "identity_sha256": identity * 64,
        "source_question_sha256": "d" * 64,
        "split": split,
        "family": "chain_sum",
        "record_count": 2,
        "question": "Compile this task.\n\nTASK:\nCompute 8 - 3 + 5.",
        "response": GOLD,
    }


def test_fault_and_exact_generic_repair():
    fault, address, corrupted = fault_ledger(GOLD, "a" * 64)
    assert address == 0
    assert corrupted == "R0|SUB|8|3|6"
    assert apply_edit_script(fault, replace_script(0, "R0|SUB|8|3|5")) == GOLD
    assert apply_edit_script(GOLD, keep_script()) == GOLD


def test_executor_rejects_wrong_address():
    with pytest.raises(LedgerEditPairError, match="outside draft"):
        apply_edit_script(GOLD, replace_script(9, "R9|ADD|1|1|2"))


def test_build_emits_balanced_pairs(tmp_path):
    train = tmp_path / "train.jsonl"
    development = tmp_path / "development.jsonl"
    train.write_text(json.dumps(row("a", "train")) + "\n")
    development.write_text(json.dumps(row("b", "development")) + "\n")
    report = build(train, development, tmp_path / "out")
    assert report["exact_fault_repair_verified"] is True
    assert report["counters"]["train"]["pairs"] == 1
    assert report["counters"]["train"]["presentations"] == 2
    assert report["record_copy_fraction"]["train"] == 0.75
    rows = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    assert [item["presentation"] for item in rows] == ["clean", "fault"]
    assert rows[0]["pair_identity_sha256"] == rows[1]["pair_identity_sha256"]


def test_depth_one_is_excluded(tmp_path):
    shallow = row("a", "train")
    shallow["record_count"] = 1
    shallow["response"] = "<LEDGER_V1>\nR0|ADD|2|3|5\nCOMMIT|@R0|5\n</LEDGER_V1>"
    train = tmp_path / "train.jsonl"
    development = tmp_path / "development.jsonl"
    train.write_text(json.dumps(shallow) + "\n")
    development.write_text(json.dumps(row("b", "development")) + "\n")
    report = build(train, development, tmp_path / "out")
    assert report["counters"]["train"]["excluded_depth_one"] == 1
    assert report["outputs"]["train"]["rows"] == 0
