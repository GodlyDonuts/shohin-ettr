from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from score_q36_mtr import build_publication_analysis
from synthesize_q36_causal_transfer import Q36CausalTransferError, synthesize

ROOT = Path(__file__).resolve().parents[1]
PRIORS = {
    "dset1": ROOT / "docs/research/SHOHIN_DSET1_RESULT.json",
    "iset1": ROOT / "docs/research/SHOHIN_ISET1_RESULT.json",
    "q35_trained": ROOT / "docs/research/SHOHIN_DSET_Q35_TRAINED_TRANSFER_RESULT.json",
}


def _q36_terminal(*, passed: bool = True, draft_supported: bool = True) -> dict:
    rows = []
    for index in range(1289):
        task = "math500" if index < 600 else "bbh_logic" if index < 1089 else "mbpp"
        rows.append(
            {
                "identity_sha256": hashlib.sha256(
                    f"synthesis-{index}".encode()
                ).hexdigest(),
                "task": task,
                "correct": {
                    "revision": index < (600 if draft_supported else 400),
                    "draft_hidden": index < 500,
                    "unchanged": index < 450,
                    "self_refinement": index < 470,
                    "learned_commit": index < 650,
                },
            }
        )
    publication = build_publication_analysis(rows)
    return {
        "schema": "shohin-q36-mtr-final-comparison-v1",
        "status": "complete",
        "run_id": "q36_synthesis_test_r1",
        "model_revision": "Qwen/Qwen3.6-35B-A3B@test",
        "formal_result": "PASS" if passed else "FAIL",
        "gate_pass": passed,
        "publication_analysis": publication,
        "publication_analysis_non_gating": True,
        "stop_after_gate": True,
        "automatic_retry_authorized": False,
        "automatic_confirmation_authorized": False,
        "automatic_successor_authorized": False,
        "next_action": "stop_and_preserve_evidence",
        "claim_boundary": "source_disjoint_development_mechanism_only",
    }


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _args(tmp_path: Path, terminal: dict | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        q36_terminal=_write(tmp_path / "q36.json", terminal or _q36_terminal()),
        dset1=PRIORS["dset1"],
        iset1=PRIORS["iset1"],
        q35_trained=PRIORS["q35_trained"],
        output=tmp_path / "synthesis.json",
    )


def test_supported_transfer_is_non_gating_and_atomic(tmp_path: Path) -> None:
    args = _args(tmp_path)
    result = synthesize(args)
    assert result["synthesis"] == {
        "prior_draft_information_effects_all_positive": True,
        "broad_task_draft_information_mechanism_transfer_supported": True,
        "dense_to_moe_architecture_pattern_replication_supported": True,
        "action_selection_bottleneck_consistent_with_prior_causal_boundary": True,
        "formal_q36_gate_result_preserved": "PASS",
    }
    assert (
        result["prior_mechanism_evidence"]["dset1"]["draft_visible_minus_hidden"] == 916
    )
    assert (
        result["prior_mechanism_evidence"]["iset1"]["draft_visible_minus_hidden"] == 582
    )
    assert (
        result["prior_mechanism_evidence"]["q35_trained_transfer"][
            "draft_visible_minus_hidden"
        ]
        == 648
    )
    assert (
        result["prior_mechanism_evidence"]["action_selection"][
            "forced_action_execution_accuracy"
        ]
        == 1.0
    )
    assert result["contract"]["q36_gate_modified"] is False
    assert result["contract"]["new_scientific_job_authorized"] is False
    assert json.loads(args.output.read_text()) == result
    with pytest.raises(Q36CausalTransferError, match="refusing existing"):
        synthesize(args)


def test_formal_fail_is_preserved_even_with_mechanism_support(tmp_path: Path) -> None:
    result = synthesize(_args(tmp_path, _q36_terminal(passed=False)))
    assert result["q36_terminal"]["formal_result"] == "FAIL"
    assert (
        result["synthesis"]["broad_task_draft_information_mechanism_transfer_supported"]
        is True
    )
    assert (
        result["synthesis"]["dense_to_moe_architecture_pattern_replication_supported"]
        is False
    )


def test_unsupported_q36_draft_claim_does_not_inherit_prior_effect(
    tmp_path: Path,
) -> None:
    result = synthesize(
        _args(tmp_path, _q36_terminal(passed=False, draft_supported=False))
    )
    assert result["q36_terminal"]["draft_visibility_claim_supported"] is False
    assert (
        result["synthesis"]["broad_task_draft_information_mechanism_transfer_supported"]
        is False
    )


def test_prior_byte_tamper_fails_closed(tmp_path: Path) -> None:
    args = _args(tmp_path)
    tampered = json.loads(PRIORS["dset1"].read_text())
    tampered["metrics"]["aligned"]["execution_correct"] -= 1
    args.dset1 = _write(tmp_path / "tampered-dset1.json", tampered)
    with pytest.raises(Q36CausalTransferError, match="hash differs"):
        synthesize(args)
    assert not args.output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"stop_after_gate": False}),
        lambda value: value.update({"automatic_successor_authorized": True}),
        lambda value: value.update({"formal_result": "PASS", "gate_pass": False}),
        lambda value: value["publication_analysis"].update(
            {"cross_board_absolute_score_comparison_authorized": True}
        ),
    ],
)
def test_terminal_contract_tamper_fails_closed(tmp_path: Path, mutation) -> None:
    terminal = _q36_terminal()
    mutation(terminal)
    args = _args(tmp_path, terminal)
    with pytest.raises(Q36CausalTransferError):
        synthesize(args)
    assert not args.output.exists()


def test_deterministic_payload_except_output_path(tmp_path: Path) -> None:
    first_args = _args(tmp_path / "first")
    first = synthesize(first_args)
    second_args = _args(tmp_path / "second")
    second = synthesize(second_args)
    for value in (first, second):
        for receipt in value["inputs"].values():
            receipt.pop("path")
    assert first == second
