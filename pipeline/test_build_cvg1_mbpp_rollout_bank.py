import hashlib
import json
from pathlib import Path

import pytest

from build_cvg1_mbpp_rollout_bank import CVG1MBPPBankError, build_bank


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(question: str, task_id: int) -> dict:
    return {
        "task": "mbpp",
        "task_id": task_id,
        "text": question,
        "code": "def solve(x):\n    return x + 1",
        "test_list": ["assert solve(2) == 3"],
        "test_setup_code": "",
        "source": "mbpp_train",
        "reference_execution_sha256": hashlib.sha256(question.encode()).hexdigest(),
    }


def test_builds_deterministic_source_disjoint_bank(tmp_path: Path) -> None:
    exact = "Return one more than the supplied integer."
    semantic = (
        "Compute a stable sequence from thirteen distinct lexical tokens now for this "
        "exact deterministic test."
    )
    kept = "Count vowels in a lowercase word without using external packages."
    source = tmp_path / "source.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    output = tmp_path / "bank.jsonl"
    report = tmp_path / "report.json"
    _write(source, [_row(exact, 1), _row(semantic, 2), _row(kept, 3)])
    _write(
        excluded,
        [
            {"question": exact},
            {
                "question": (
                    "Compute a stable sequence from thirteen distinct lexical tokens "
                    "now for this exact deterministic test with an added suffix."
                )
            },
        ],
    )

    receipt = build_bank(
        source,
        [excluded],
        output,
        report,
        count=1,
        seed=17,
        max_reference_ngram_document_rate=0.0,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["text"] for row in rows] == [kept]
    assert rows[0]["task"] == "mbpp"
    assert receipt["rows"] == 1
    assert receipt["counters"]["excluded_overlap"] == 1
    assert receipt["counters"]["excluded_informative_ngram_overlap"] == 1


def test_fails_closed_when_admissible_count_is_too_small(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    _write(source, [_row("Increment an integer.", 1)])
    _write(excluded, [{"question": "Increment an integer!"}])

    with pytest.raises(CVG1MBPPBankError, match="below requested"):
        build_bank(
            source,
            [excluded],
            tmp_path / "bank.jsonl",
            tmp_path / "report.json",
            count=1,
            seed=17,
        )


def test_rejects_unverified_source_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    row = _row("Increment an integer.", 1)
    row["reference_execution_sha256"] = ""
    _write(source, [row])
    _write(excluded, [])

    with pytest.raises(CVG1MBPPBankError, match="below requested"):
        build_bank(
            source,
            [excluded],
            tmp_path / "bank.jsonl",
            tmp_path / "report.json",
            count=1,
            seed=17,
        )
