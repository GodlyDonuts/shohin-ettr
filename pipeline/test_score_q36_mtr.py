from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from score_q36_mtr import (
    CONSUMPTION_SCHEMA,
    Q36MTRScoreError,
    SELECTION_SCHEMA,
    TERMINAL_FAILURE_SCHEMA,
    _load_selections,
    _preserve_post_consumption_failure,
)


def _selections(path: Path) -> None:
    rows = [
        {
            "schema": SELECTION_SCHEMA,
            "identity_sha256": hashlib.sha256(
                f"selection-{index}".encode()
            ).hexdigest(),
            "task": ("math500", "bbh_logic", "mbpp")[index % 3],
            "selected_index": index % 2,
            "selected_lineage": ("revision", "unchanged")[index % 2],
            "order_consistent": True,
            "margin": float(index),
        }
        for index in range(1_289)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_q36_selection_loader_has_exact_order_symmetric_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selections.jsonl"
    _selections(path)
    assert len(_load_selections(path)) == 1_289


def test_q36_selection_loader_rejects_lineage_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "selections.jsonl"
    _selections(path)
    rows = path.read_text().splitlines()
    first = json.loads(rows[0])
    first["selected_lineage"] = "unchanged"
    rows[0] = json.dumps(first)
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(Q36MTRScoreError):
        _load_selections(path)


def test_q36_scorer_claims_before_sole_board_open() -> None:
    source = Path(__file__).with_name("score_q36_mtr.py").read_text()
    assert source.index("consumption_sha256 = _consume(") < source.index(
        "assessors, observed_board_sha256 = _load_assessors_once("
    )
    assert 'assessor_semantic_reads": 1' in source


def test_q36_post_consumption_failure_is_terminal_and_nonretryable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "score"
    authorization = tmp_path / "authorization.json"
    authorization.write_text('{"schema":"authorization"}\n', encoding="utf-8")
    authorization_sha256 = hashlib.sha256(authorization.read_bytes()).hexdigest()
    consumption = tmp_path / "score.score-authorization-consumed.json"
    consumption.write_text(
        json.dumps(
            {
                "schema": CONSUMPTION_SCHEMA,
                "status": "consumed",
                "run_id": "q36-test",
                "authorization_sha256": authorization_sha256,
                "score_output_root": str(output.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {"output": output, "score_authorization": authorization},
    )()
    _preserve_post_consumption_failure(args, RuntimeError("scoring failed"))
    failure = json.loads(
        (tmp_path / "score.terminal-failure.json").read_text(encoding="utf-8")
    )
    assert failure["schema"] == TERMINAL_FAILURE_SCHEMA
    assert failure["score_consumption_state"] == "consumed"
    assert failure["assessor_semantic_read_state"] == "zero_or_partial_unknown"
    assert failure["retry_authorized"] is False
    assert failure["successor_authorized"] is False
