import hashlib
import json
from pathlib import Path

from train_dseo1_paired_action import load_paired_rows, paired_order


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_paired_rows_requires_fixed_source_and_final(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    rows = []
    for member in ("clean", "fault"):
        rows.append(
            {
                "schema": "shohin-dseo1-paired-presentation-v1",
                "pair_identity_sha256": "pair",
                "pair_member": member,
                "source_identity_sha256": "source",
                "final_response": "answer",
            }
        )
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "shohin-dseo1-paired-data-report-v1",
                "status": "complete",
                "holdout_used": False,
                "pair_balance_exact": True,
                "train_diagnostic_source_overlap": 0,
                "complete_retention": True,
                "max_sequence_length": 4096,
                "outputs": {
                    "train": {
                        "path": str(data.resolve()),
                        "sha256": _sha(data),
                        "sources": 1,
                    }
                },
            }
        )
    )
    pairs, _ = load_paired_rows(data, report)
    assert len(pairs) == 1
    assert [row["pair_member"] for row in pairs[0]] == ["clean", "fault"]


def test_paired_order_is_reproducible() -> None:
    pairs = [[index] for index in range(20)]
    assert paired_order(pairs, 17) == paired_order(pairs, 17)
    assert paired_order(pairs, 17) != paired_order(pairs, 18)
