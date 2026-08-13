#!/usr/bin/env python3
"""Synthesize terminal Q36 evidence with the frozen DSET/ISET causal boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from score_q36_mtr import Q36MTRScoreError, validate_publication_analysis

OUTPUT_SCHEMA = "shohin-q36-causal-transfer-synthesis-v1"
Q36_SCHEMA = "shohin-q36-mtr-final-comparison-v1"
PRIOR_INPUTS = {
    "dset1": {
        "schema": "shohin-dset1-result-v1",
        "sha256": "7915c92475992dd95c8d6b2f2073699130b97cb63c2e323eff4a780cad55c44a",
    },
    "iset1": {
        "schema": "shohin-iset1-stage0-comparison-v1",
        "sha256": "b62897ac2044518f872977c4cdb3f7b2409524a5435030180b2bd559d7815f5d",
    },
    "q35_trained": {
        "schema": "shohin-dset-q35-trained-transfer-result-v1",
        "sha256": "c0452f9af07e44b5f8d69fc0971fd0dca6a96c3385a6aad74488d26d3606bfe9",
    },
}


class Q36CausalTransferError(RuntimeError):
    """The supplied evidence cannot support the causal-transfer synthesis."""


def _load(path: Path, schema: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q36CausalTransferError(f"unreadable {label}: {path}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("status") != "complete"
    ):
        raise Q36CausalTransferError(f"incomplete {label}: {path}")
    return value, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _load_prior(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = PRIOR_INPUTS[label]
    value, receipt = _load(path, contract["schema"], label)
    if receipt["sha256"] != contract["sha256"]:
        raise Q36CausalTransferError(f"{label} evidence hash differs")
    return value, receipt


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Q36CausalTransferError(f"invalid {label}")
    return value


def _positive_effect(
    *, rows: int, treatment: int, hidden: int, swapped: int | None = None
) -> dict[str, Any]:
    if rows <= 0 or treatment > rows or hidden > rows:
        raise Q36CausalTransferError("causal-boundary count differs")
    if swapped is not None and swapped > rows:
        raise Q36CausalTransferError("causal-boundary swapped count differs")
    margin = treatment - hidden
    result: dict[str, Any] = {
        "rows": rows,
        "draft_visible_correct": treatment,
        "draft_hidden_correct": hidden,
        "draft_visible_minus_hidden": margin,
        "draft_visible_minus_hidden_percentage_points": margin * 100.0 / rows,
        "positive_draft_information_effect": margin > 0,
    }
    if swapped is not None:
        result.update(
            {
                "draft_swapped_correct": swapped,
                "draft_visible_minus_swapped": treatment - swapped,
                "positive_binding_effect": treatment > swapped,
            }
        )
    return result


def _validate_q36(value: dict[str, Any]) -> dict[str, Any]:
    if (
        value.get("formal_result") not in {"PASS", "FAIL"}
        or not isinstance(value.get("gate_pass"), bool)
        or (value["formal_result"] == "PASS") is not value["gate_pass"]
        or value.get("stop_after_gate") is not True
        or value.get("automatic_retry_authorized") is not False
        or value.get("automatic_confirmation_authorized") is not False
        or value.get("automatic_successor_authorized") is not False
        or value.get("next_action") != "stop_and_preserve_evidence"
        or value.get("claim_boundary") != "source_disjoint_development_mechanism_only"
    ):
        raise Q36CausalTransferError("Q36 terminal contract differs")
    publication = value.get("publication_analysis")
    try:
        publication = validate_publication_analysis(publication)
    except Q36MTRScoreError as error:
        raise Q36CausalTransferError("Q36 publication analysis differs") from error
    if (
        value.get("publication_analysis_non_gating") is not True
        or publication.get("status") != "descriptive_non_gating"
        or publication.get("cross_board_absolute_score_comparison_authorized")
        is not False
        or publication.get("gate_thresholds_modified") is not False
        or publication.get("automatic_successor_authorized") is not False
    ):
        raise Q36CausalTransferError("Q36 publication boundary differs")
    evidence = publication.get("claim_evidence")
    comparisons = publication.get("comparisons")
    if not isinstance(evidence, dict) or not isinstance(comparisons, dict):
        raise Q36CausalTransferError("Q36 publication evidence is missing")
    claim = evidence.get("claims", {}).get("draft_visibility_causal")
    comparison = comparisons.get("revision_vs_draft_hidden")
    if (
        not isinstance(claim, dict)
        or not isinstance(comparison, dict)
        or claim.get("comparison") != "revision_vs_draft_hidden"
        or not isinstance(claim.get("publication_claim_supported"), bool)
        or not isinstance(evidence.get("draft_visibility_causal_supported"), bool)
        or evidence.get("draft_visibility_causal_supported")
        is not claim.get("publication_claim_supported")
    ):
        raise Q36CausalTransferError("Q36 draft-visibility evidence differs")
    overall = comparison.get("overall")
    if not isinstance(overall, dict):
        raise Q36CausalTransferError("Q36 paired evidence is missing")
    numeric = (
        "rows",
        "treatment_correct",
        "control_correct",
        "net_correct",
        "risk_difference_percentage_points",
    )
    if any(
        isinstance(overall.get(field), bool)
        or not isinstance(overall.get(field), (int, float))
        or not math.isfinite(float(overall[field]))
        for field in numeric
    ):
        raise Q36CausalTransferError("Q36 paired evidence differs")
    if (
        overall["rows"] <= 0
        or overall["net_correct"]
        != overall["treatment_correct"] - overall["control_correct"]
    ):
        raise Q36CausalTransferError("Q36 paired arithmetic differs")
    return {
        "formal_result": value["formal_result"],
        "gate_pass": value["gate_pass"],
        "run_id": value.get("run_id"),
        "model_revision": value.get("model_revision"),
        "draft_visibility_claim_supported": evidence[
            "draft_visibility_causal_supported"
        ],
        "dense_pattern_replication_supported": evidence.get(
            "dense_pattern_replication_supported"
        )
        is True,
        "revision_vs_draft_hidden": overall,
        "holm_claim": claim,
    }


def _validate_priors(
    dset1: dict[str, Any], iset1: dict[str, Any], q35: dict[str, Any]
) -> dict[str, Any]:
    dset_metrics = dset1.get("metrics")
    if not isinstance(dset_metrics, dict):
        raise Q36CausalTransferError("DSET1 metrics are missing")
    dset = _positive_effect(
        rows=_integer(dset1.get("data", {}).get("diagnostic_rows"), "DSET1 rows"),
        treatment=_integer(
            dset_metrics.get("aligned", {}).get("execution_correct"),
            "DSET1 aligned",
        ),
        hidden=_integer(
            dset_metrics.get("hidden", {}).get("execution_correct"), "DSET1 hidden"
        ),
        swapped=_integer(
            dset_metrics.get("swapped", {}).get("execution_correct"),
            "DSET1 swapped",
        ),
    )
    if (
        dset_metrics.get("margins", {}).get("aligned_minus_hidden_answers")
        != dset["draft_visible_minus_hidden"]
        or dset_metrics.get("margins", {}).get("aligned_minus_swapped_answers")
        != dset["draft_visible_minus_swapped"]
    ):
        raise Q36CausalTransferError("DSET1 margin binding differs")

    iset_metrics = iset1.get("metrics")
    if not isinstance(iset_metrics, dict):
        raise Q36CausalTransferError("ISET1 metrics are missing")
    iset = _positive_effect(
        rows=_integer(iset1.get("row_count"), "ISET1 rows"),
        treatment=_integer(
            iset_metrics.get("aligned", {}).get("execution_correct"),
            "ISET1 aligned",
        ),
        hidden=_integer(
            iset_metrics.get("hidden", {}).get("execution_correct"), "ISET1 hidden"
        ),
        swapped=_integer(
            iset_metrics.get("swapped", {}).get("execution_correct"),
            "ISET1 swapped",
        ),
    )
    if (
        iset1.get("margins", {}).get("aligned_minus_hidden")
        != iset["draft_visible_minus_hidden"]
        or iset1.get("margins", {}).get("aligned_minus_swapped")
        != iset["draft_visible_minus_swapped"]
    ):
        raise Q36CausalTransferError("ISET1 margin binding differs")

    development = q35.get("development")
    forced = q35.get("forced_action_attribution")
    if not isinstance(development, dict) or not isinstance(forced, dict):
        raise Q36CausalTransferError("Q35 trained evidence is missing")
    q35_effect = _positive_effect(
        rows=_integer(development.get("rows"), "Q35 rows"),
        treatment=_integer(development.get("aligned_execution_correct"), "Q35 aligned"),
        hidden=_integer(development.get("hidden_execution_correct"), "Q35 hidden"),
    )
    if (
        development.get("aligned_minus_hidden")
        != q35_effect["draft_visible_minus_hidden"]
    ):
        raise Q36CausalTransferError("Q35 margin binding differs")
    choice_exact = _integer(development.get("choice_script_exact"), "Q35 choice")
    choice_rows = _integer(development.get("choice_rows"), "Q35 choice rows")
    forced_script = _integer(forced.get("script_exact"), "Q35 forced script")
    forced_execution = _integer(forced.get("execution_correct"), "Q35 forced execution")
    if (
        choice_rows != 256
        or choice_exact != 177
        or forced.get("scope") != "all 128 faulted choice rows"
        or forced_script != 128
        or forced_execution != 128
    ):
        raise Q36CausalTransferError("Q35 action-selection boundary differs")
    return {
        "dset1": dset,
        "iset1": iset,
        "q35_trained_transfer": q35_effect,
        "action_selection": {
            "natural_choice_script_exact": choice_exact,
            "natural_choice_rows": choice_rows,
            "natural_choice_script_accuracy": choice_exact / choice_rows,
            "forced_action_script_exact": forced_script,
            "forced_action_execution_correct": forced_execution,
            "forced_action_rows": 128,
            "forced_action_execution_accuracy": forced_execution / 128,
            "value_generator_succeeds_when_action_is_fixed": True,
            "measured_bottleneck": "model_owned_action_selection",
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36CausalTransferError(f"refusing existing synthesis: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise Q36CausalTransferError(f"refusing existing synthesis: {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def synthesize(args: argparse.Namespace) -> dict[str, Any]:
    q36, q36_receipt = _load(Path(args.q36_terminal), Q36_SCHEMA, "Q36 terminal")
    dset1, dset1_receipt = _load_prior(Path(args.dset1), "dset1")
    iset1, iset1_receipt = _load_prior(Path(args.iset1), "iset1")
    q35, q35_receipt = _load_prior(Path(args.q35_trained), "q35_trained")
    q36_summary = _validate_q36(q36)
    priors = _validate_priors(dset1, iset1, q35)
    prior_effects_positive = all(
        priors[name]["positive_draft_information_effect"]
        for name in ("dset1", "iset1", "q35_trained_transfer")
    )
    mechanism_transfer = (
        q36_summary["draft_visibility_claim_supported"] and prior_effects_positive
    )
    architecture_transfer = (
        q36_summary["gate_pass"] and q36_summary["dense_pattern_replication_supported"]
    )
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete_non_gating",
        "inputs": {
            "q36_terminal": q36_receipt,
            "dset1": dset1_receipt,
            "iset1": iset1_receipt,
            "q35_trained": q35_receipt,
        },
        "q36_terminal": q36_summary,
        "prior_mechanism_evidence": priors,
        "synthesis": {
            "prior_draft_information_effects_all_positive": prior_effects_positive,
            "broad_task_draft_information_mechanism_transfer_supported": mechanism_transfer,
            "dense_to_moe_architecture_pattern_replication_supported": architecture_transfer,
            "action_selection_bottleneck_consistent_with_prior_causal_boundary": True,
            "formal_q36_gate_result_preserved": q36_summary["formal_result"],
        },
        "interpretation": {
            "supported_if_mechanism_transfer_true": (
                "Draft-visible model-owned revision carries causally useful information "
                "across the prior synthetic edit boundary and the source-disjoint broad "
                "Q36 board."
            ),
            "supported_action_boundary": (
                "Prior forced-action evidence isolates model-owned action selection as "
                "the measured bottleneck; it does not establish a universal bottleneck."
            ),
            "architecture_claim_requires_q36_pass": True,
            "statistical_pooling_across_boards_performed": False,
            "cross_board_absolute_score_comparison_authorized": False,
            "scaling_law_claim_authorized": False,
        },
        "contract": {
            "q36_gate_modified": False,
            "non_gating_post_terminal_analysis": True,
            "automatic_retry_authorized": False,
            "automatic_confirmation_authorized": False,
            "automatic_successor_authorized": False,
            "new_scientific_job_authorized": False,
            "next_action": "preserve_and_report_only",
        },
    }
    _atomic_json(Path(args.output), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q36-terminal", required=True)
    parser.add_argument("--dset1", required=True)
    parser.add_argument("--iset1", required=True)
    parser.add_argument("--q35-trained", required=True)
    parser.add_argument("--output", required=True)
    result = synthesize(parser.parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
