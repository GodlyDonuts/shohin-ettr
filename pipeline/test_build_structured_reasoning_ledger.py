import json

import pytest

from build_structured_reasoning_ledger import (
    StructuredLedgerError,
    build,
    compile_response,
)


def test_compile_response_builds_exact_dependencies():
    records = compile_response(
        "<think>8.347 - 5.087 = 3.26 ; 4.665 - 3.26 = 1.405 ; "
        "1.405 * 6.571 = 9.232255</think>\nThe answer is 9.232255.",
        "9.232255",
    )
    assert [record["operation"] for record in records] == ["SUB", "SUB", "MUL"]
    assert records[1]["dependencies"] == [{"operand_role": "right", "record_index": 0}]
    assert records[2]["dependencies"] == [{"operand_role": "left", "record_index": 1}]
    assert records[-1]["result"] == {"numerator": 1846451, "denominator": 200000}


def test_compile_response_rejects_false_step():
    with pytest.raises(StructuredLedgerError, match="clause arithmetic is false"):
        compile_response("<think>2 + 2 = 5</think>\nThe answer is 5.", "5")


def test_build_writes_disjoint_exact_ledgers(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = []
    for index in range(40):
        value = index + 3
        rows.append(
            {
                "question": f"Compute row {index}: {value} + 2",
                "response": f"<think>{value} + 2 = {value + 2}</think>\nThe answer is {value + 2}.",
                "answer": str(value + 2),
                "family": "chain_sum",
            }
        )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = build(source, tmp_path / "output", development_modulus=3)

    assert report["exact_arithmetic_verified"] is True
    assert report["holdout_used"] is False
    assert report["counters"]["admitted_rows"] == 40
    assert report["outputs"]["train"]["rows"] > 0
    assert report["outputs"]["development"]["rows"] > 0
    identities = set()
    for split in ("train", "development"):
        for line in (tmp_path / "output" / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["identity_sha256"] not in identities
            identities.add(row["identity_sha256"])


def test_build_rejects_source_hash_drift(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("")
    with pytest.raises(StructuredLedgerError, match="source SHA-256 differs"):
        build(source, tmp_path / "output", expected_source_sha256="0" * 64)
