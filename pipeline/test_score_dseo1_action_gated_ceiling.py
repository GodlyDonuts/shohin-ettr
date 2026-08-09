import json

import pytest

from score_dseo1_action_gated_ceiling import DSEC0Error, load_report, score


def _row(identity, pair, member, action, answer, family="numeric_final"):
    return {
        "identity_sha256": identity,
        "pair_identity_sha256": pair,
        "pair_member": member,
        "corruption_family": family,
        "gold_answer": "1",
        "predicted_action": action,
        "answer_correct": answer,
    }


def test_action_gate_copies_keep_and_generates_fix() -> None:
    aligned = {
        "results": [
            _row("c", "p", "clean", "<KEEP>", False),
            _row("f", "p", "fault", "<FIX_FINAL>", True),
        ]
    }
    final_only = {
        "results": [
            _row("c", "p", "clean", None, False),
            _row("f", "p", "fault", None, False),
        ]
    }
    report = score(aligned, final_only)
    assert report["metrics"]["observed_action_gated"]["correct"] == 2
    assert report["metrics"]["oracle_action_gated"]["correct"] == 2
    assert report["metrics"]["final_only"]["correct"] == 0


def test_wrong_keep_copies_fault_and_fails() -> None:
    aligned = {
        "results": [
            _row("c", "p", "clean", "<FIX_FINAL>", True),
            _row("f", "p", "fault", "<KEEP>", True),
        ]
    }
    final_only = {
        "results": [
            _row("c", "p", "clean", None, True),
            _row("f", "p", "fault", None, True),
        ]
    }
    report = score(aligned, final_only)
    assert report["metrics"]["observed_action_gated"]["correct"] == 1
    assert report["metrics"]["oracle_action_gated"]["correct"] == 2


def test_load_report_accepts_registered_merge_but_rejects_holdout(tmp_path) -> None:
    path = tmp_path / "report.json"
    report = {
        "schema": "shohin-dseo1-paired-evaluation-merged-v1",
        "status": "complete",
        "arm": "aligned",
        "row_count": 0,
        "results": [],
    }
    path.write_text(json.dumps(report))
    assert load_report(path, "aligned") == report
    report["holdout_used"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(DSEC0Error):
        load_report(path, "aligned")
