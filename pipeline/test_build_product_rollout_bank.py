import json
from pathlib import Path

import pytest

from pipeline.build_product_rollout_bank import (
    ProductRolloutBankError,
    build_bank,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_build_bank_is_fresh_balanced_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    rows = []
    for group in ("math", "science"):
        for index in range(5):
            rows.append(
                {
                    "question": f"{group} question {index}",
                    "expected_answer_normalized": str(index),
                    "training_group": group,
                    "verification": "expected_answer_match_v1",
                    "prompt_sha256": f"source-{group}-{index}",
                }
            )
    _write(source, rows)
    _write(excluded, [{"question": "math question 0"}])
    output = tmp_path / "bank.jsonl"
    report_path = tmp_path / "report.json"
    report = build_bank(
        [source],
        [excluded],
        output,
        report_path,
        counts={"math": 3, "science": 2},
        seed=7,
    )
    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["counts_selected"] == {"math": 3, "science": 2}
    assert len(selected) == 5
    assert len({row["identity_sha256"] for row in selected}) == 5
    assert all(row["question"] != "math question 0" for row in selected)
    assert {row["task"] for row in selected} == {"math500", "bbh_logic"}
    assert all(row["answer"].startswith(r"\boxed{") for row in selected)


def test_build_bank_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write(
        source,
        [
            {
                "question": "q",
                "expected_answer_normalized": "1",
                "training_group": "math",
                "verification": "expected_answer_match_v1",
            }
        ],
    )
    output = tmp_path / "bank.jsonl"
    output.write_text("occupied")
    with pytest.raises(ProductRolloutBankError, match="refusing"):
        build_bank(
            [source],
            [],
            output,
            tmp_path / "report.json",
            counts={"math": 1},
            seed=1,
        )
