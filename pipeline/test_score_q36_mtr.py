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
    build_publication_analysis,
    _load_selections,
    _mcnemar_exact,
    _paired_summary_from_counts,
    _preserve_post_consumption_failure,
)


def _publication_outcomes() -> list[dict]:
    task_counts = {"math500": 600, "bbh_logic": 489, "mbpp": 200}
    rows = []
    index = 0
    for task, count in task_counts.items():
        for offset in range(count):
            rows.append(
                {
                    "identity_sha256": hashlib.sha256(
                        f"publication-{index}".encode()
                    ).hexdigest(),
                    "task": task,
                    "correct": {
                        "revision": offset % 5 != 0,
                        "unchanged": offset % 3 == 0,
                        "self_refinement": offset % 4 == 0,
                        "draft_hidden": offset % 7 == 0,
                        "learned_commit": offset % 6 != 0,
                    },
                }
            )
            index += 1
    return rows


def test_q36_publication_analysis_is_paired_exact_and_non_gating() -> None:
    report = build_publication_analysis(_publication_outcomes())
    comparison = report["comparisons"]["revision_vs_draft_hidden"]
    assert comparison["overall"]["rows"] == 1_289
    assert comparison["overall"]["net_correct"] == (
        comparison["overall"]["treatment_correct"]
        - comparison["overall"]["control_correct"]
    )
    assert set(comparison["domains"]) == {"math500", "bbh_logic", "mbpp"}
    assert report["gate_fields_read"] is False
    assert report["gate_thresholds_modified"] is False
    assert report["cross_board_absolute_score_comparison_authorized"] is False
    claims = report["claim_evidence"]
    assert claims["multiple_comparison_method"] == (
        "holm_bonferroni_exact_mcnemar_family"
    )
    assert claims["gate_thresholds_modified"] is False
    assert claims["draft_visibility_causal_supported"] is True


def test_q36_publication_exact_probability_and_paired_interval_are_frozen() -> None:
    assert _mcnemar_exact(3, 0) == {
        "method": "exact_two_sided_mcnemar_binomial",
        "numerator": "1",
        "denominator": "4",
        "value": 0.25,
    }
    summary = _paired_summary_from_counts(3, 0, 5, 2)
    assert summary["rows"] == 10
    assert summary["net_correct"] == 3
    assert summary["risk_difference_percentage_points"] == 30.0
    assert summary["paired_wald_95_ci_percentage_points"][0] < 30.0
    assert summary["paired_wald_95_ci_percentage_points"][1] > 30.0


def test_q36_publication_claims_remain_unsupported_without_paired_effect() -> None:
    outcomes = _publication_outcomes()
    for row in outcomes:
        row["correct"] = {arm: True for arm in row["correct"]}
    claims = build_publication_analysis(outcomes)["claim_evidence"]
    assert claims["draft_visibility_causal_supported"] is False
    assert claims["dense_pattern_replication_supported"] is False
    assert not any(
        value["publication_claim_supported"] for value in claims["claims"].values()
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
