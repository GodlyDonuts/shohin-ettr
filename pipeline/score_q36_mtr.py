#!/usr/bin/env python3
"""Consume one Q36 score authorization and open the development board once."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from build_pcf1_data import (
    CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA,
    CONFIRMATION_ASSESSOR_SCHEMA,
)
from build_q36_mtr_commit_pairs import _load_arm, sha256_file
from hf_q36_mtr_evaluate import TASKS, load_rows
from pcf1_code_sandbox import (
    BWRAP_SHA256,
    SANDBOX_CONFIG_SHA256,
    atomic_json as sandbox_atomic_json,
    mbpp_allocation_setup_receipts_sha256,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
    score_completion,
)
from q36_mtr_roles import MODEL_REVISION

AUTHORIZATION_SCHEMA = "shohin-q36-mtr-score-authorization-v1"
CONSUMPTION_SCHEMA = "shohin-q36-mtr-score-consumption-v1"
SELECTION_SCHEMA = "shohin-q36-mtr-commit-selection-v1"
APPLICATION_SCHEMA = "shohin-q36-mtr-commit-application-report-v1"
COMMIT_REPORT_SCHEMA = "shohin-q36-mtr-commit-training-report-v1"
SCORE_SCHEMA = "shohin-q36-mtr-score-result-v1"
OUTCOME_SCHEMA = "shohin-q36-mtr-scored-outcome-v1"
TERMINAL_FAILURE_SCHEMA = "shohin-q36-mtr-score-terminal-failure-v1"
PUBLICATION_ANALYSIS_SCHEMA = "shohin-q36-mtr-publication-analysis-v1"
ARMS = ("revision", "unchanged", "self_refinement", "draft_hidden")
TOTAL_ROWS = 1_289
PUBLICATION_COMPARISONS = {
    "revision_vs_unchanged": ("revision", "unchanged"),
    "revision_vs_self_refinement": ("revision", "self_refinement"),
    "revision_vs_draft_hidden": ("revision", "draft_hidden"),
    "learned_commit_vs_revision": ("learned_commit", "revision"),
    "learned_commit_vs_unchanged": ("learned_commit", "unchanged"),
}
DENSE_REFERENCE = {
    "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "board": "qualified_product_538",
    "scores": {"unchanged": 316, "trained_revision": 374, "learned_commit": 383},
    "total": 538,
    "revision_minus_unchanged": 58,
    "commit_minus_revision": 9,
}
Q36_REFERENCE = {
    "model": "Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0",
    "board": "source_disjoint_development_1289",
    "total": TOTAL_ROWS,
}


class Q36MTRScoreError(RuntimeError):
    """The Q36 one-open score boundary or its inputs differ."""


def _mcnemar_exact(wins: int, losses: int) -> dict[str, Any]:
    discordant = wins + losses
    if discordant == 0:
        probability = Fraction(1, 1)
    else:
        tail = sum(
            math.comb(discordant, index) for index in range(min(wins, losses) + 1)
        )
        probability = min(Fraction(1, 1), Fraction(2 * tail, 1 << discordant))
    return {
        "method": "exact_two_sided_mcnemar_binomial",
        "numerator": str(probability.numerator),
        "denominator": str(probability.denominator),
        "value": float(probability),
    }


def _paired_summary_from_counts(
    wins: int, losses: int, both_correct: int, both_incorrect: int
) -> dict[str, Any]:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (wins, losses, both_correct, both_incorrect)
    ):
        raise Q36MTRScoreError("Q36 paired publication count differs")
    total = wins + losses + both_correct + both_incorrect
    if total <= 1:
        raise Q36MTRScoreError("Q36 paired publication cardinality differs")
    net = wins - losses
    mean = net / total
    sum_squares = wins + losses
    variance = max(0.0, (sum_squares - total * mean * mean) / (total - 1))
    standard_error = math.sqrt(variance / total)
    critical = 1.959963984540054
    lower = max(-1.0, mean - critical * standard_error)
    upper = min(1.0, mean + critical * standard_error)
    return {
        "rows": total,
        "treatment_correct": wins + both_correct,
        "control_correct": losses + both_correct,
        "treatment_only_correct": wins,
        "control_only_correct": losses,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "discordant": wins + losses,
        "net_correct": net,
        "risk_difference": mean,
        "risk_difference_percentage_points": mean * 100.0,
        "paired_wald_95_ci_percentage_points": [lower * 100.0, upper * 100.0],
        "paired_wald_standard_error": standard_error,
        "mcnemar_exact_two_sided": _mcnemar_exact(wins, losses),
    }


def _paired_summary(
    outcomes: list[dict[str, Any]], treatment: str, control: str
) -> dict[str, Any]:
    counts = Counter()
    for row in outcomes:
        correct = row.get("correct")
        if not isinstance(correct, dict):
            raise Q36MTRScoreError("Q36 publication correctness differs")
        treatment_correct = correct.get(treatment)
        control_correct = correct.get(control)
        if not isinstance(treatment_correct, bool) or not isinstance(
            control_correct, bool
        ):
            raise Q36MTRScoreError("Q36 publication Boolean differs")
        counts[(treatment_correct, control_correct)] += 1
    return _paired_summary_from_counts(
        counts[(True, False)],
        counts[(False, True)],
        counts[(True, True)],
        counts[(False, False)],
    )


def build_publication_analysis(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    if len(outcomes) != TOTAL_ROWS:
        raise Q36MTRScoreError("Q36 publication outcome count differs")
    identities = [row.get("identity_sha256") for row in outcomes]
    if (
        len(set(identities)) != TOTAL_ROWS
        or any(not isinstance(value, str) or len(value) != 64 for value in identities)
        or any(row.get("task") not in TASKS for row in outcomes)
    ):
        raise Q36MTRScoreError("Q36 publication identity coverage differs")
    comparisons = {}
    for name, (treatment, control) in PUBLICATION_COMPARISONS.items():
        domains = {
            task: _paired_summary(
                [row for row in outcomes if row["task"] == task], treatment, control
            )
            for task in TASKS
        }
        comparisons[name] = {
            "treatment": treatment,
            "control": control,
            "overall": _paired_summary(outcomes, treatment, control),
            "domains": domains,
        }
    payload = {
        "schema": PUBLICATION_ANALYSIS_SCHEMA,
        "status": "descriptive_non_gating",
        "rows": TOTAL_ROWS,
        "comparisons": comparisons,
        "dense_reference": DENSE_REFERENCE,
        "scaling_graph_data": _scaling_graph_data(comparisons),
        "scaling_interpretation": (
            "within_board_paired_effects_only; dense and MoE absolute scores use "
            "different source-disjoint boards"
        ),
        "cross_board_absolute_score_comparison_authorized": False,
        "gate_fields_read": False,
        "gate_thresholds_modified": False,
        "automatic_successor_authorized": False,
    }
    validate_publication_analysis(payload)
    return payload


def validate_publication_analysis(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "status",
        "rows",
        "comparisons",
        "dense_reference",
        "scaling_graph_data",
        "scaling_interpretation",
        "cross_board_absolute_score_comparison_authorized",
        "gate_fields_read",
        "gate_thresholds_modified",
        "automatic_successor_authorized",
    }:
        raise Q36MTRScoreError("Q36 publication analysis shape differs")
    comparisons = value.get("comparisons")
    if (
        value.get("schema") != PUBLICATION_ANALYSIS_SCHEMA
        or value.get("status") != "descriptive_non_gating"
        or value.get("rows") != TOTAL_ROWS
        or value.get("dense_reference") != DENSE_REFERENCE
        or value.get("scaling_interpretation")
        != (
            "within_board_paired_effects_only; dense and MoE absolute scores use "
            "different source-disjoint boards"
        )
        or value.get("cross_board_absolute_score_comparison_authorized") is not False
        or value.get("gate_fields_read") is not False
        or value.get("gate_thresholds_modified") is not False
        or value.get("automatic_successor_authorized") is not False
        or not isinstance(comparisons, dict)
        or set(comparisons) != set(PUBLICATION_COMPARISONS)
    ):
        raise Q36MTRScoreError("Q36 publication analysis contract differs")
    for name, (treatment, control) in PUBLICATION_COMPARISONS.items():
        comparison = comparisons[name]
        if (
            not isinstance(comparison, dict)
            or comparison.get("treatment") != treatment
            or comparison.get("control") != control
            or set(comparison) != {"treatment", "control", "overall", "domains"}
            or not isinstance(comparison.get("domains"), dict)
            or set(comparison["domains"]) != set(TASKS)
        ):
            raise Q36MTRScoreError("Q36 publication comparison differs")
        summaries = [comparison["overall"], *comparison["domains"].values()]
        for summary in summaries:
            if not isinstance(summary, dict):
                raise Q36MTRScoreError("Q36 publication summary differs")
            fields = (
                "treatment_only_correct",
                "control_only_correct",
                "both_correct",
                "both_incorrect",
            )
            counts = [summary.get(field) for field in fields]
            if any(
                isinstance(item, bool) or not isinstance(item, int) for item in counts
            ):
                raise Q36MTRScoreError("Q36 publication counts differ")
            if summary != _paired_summary_from_counts(*counts):
                raise Q36MTRScoreError("Q36 publication arithmetic differs")
        for field in (
            "rows",
            "treatment_correct",
            "control_correct",
            "treatment_only_correct",
            "control_only_correct",
            "both_correct",
            "both_incorrect",
            "discordant",
            "net_correct",
        ):
            if comparison["overall"][field] != sum(
                comparison["domains"][task][field] for task in TASKS
            ):
                raise Q36MTRScoreError("Q36 publication domain sum differs")
    if value.get("scaling_graph_data") != _scaling_graph_data(comparisons):
        raise Q36MTRScoreError("Q36 publication scaling graph differs")
    return value


def _scaling_graph_data(comparisons: dict[str, Any]) -> dict[str, Any]:
    revision_control = comparisons["revision_vs_unchanged"]["overall"]
    commit_revision = comparisons["learned_commit_vs_revision"]["overall"]
    revision_self = comparisons["revision_vs_self_refinement"]["overall"]
    revision_hidden = comparisons["revision_vs_draft_hidden"]["overall"]
    dense_scores = DENSE_REFERENCE["scores"]
    q36_scores = {
        "unchanged": revision_control["control_correct"],
        "trained_revision": revision_control["treatment_correct"],
        "learned_commit": commit_revision["treatment_correct"],
        "self_refinement": revision_self["control_correct"],
        "draft_hidden": revision_hidden["control_correct"],
    }
    return {
        "schema": "shohin-architecture-transfer-scaling-graph-data-v1",
        "interpretation": "two_board_architecture_transfer_not_compute_scaling_law",
        "points": [
            {
                **DENSE_REFERENCE,
                "arm_percentages": {
                    arm: score * 100.0 / DENSE_REFERENCE["total"]
                    for arm, score in dense_scores.items()
                },
            },
            {
                **Q36_REFERENCE,
                "scores": q36_scores,
                "arm_percentages": {
                    arm: score * 100.0 / TOTAL_ROWS for arm, score in q36_scores.items()
                },
            },
        ],
        "cross_board_absolute_score_comparison_authorized": False,
    }


def _load_assessors_once(
    path: Path, expected_sha256: str
) -> tuple[dict[str, dict[str, Any]], str]:
    """Read and parse the assessor artifact exactly once after consumption."""

    rows = {}
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for raw_line in handle:
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                identity = row.get("identity_sha256")
                assessor = row.get("assessor")
                if (
                    row.get("schema") != CONFIRMATION_ASSESSOR_SCHEMA
                    or row.get("split") != "confirmation"
                    or not isinstance(identity, str)
                    or len(identity) != 64
                    or identity in rows
                    or row.get("task") not in TASKS
                    or not isinstance(assessor, dict)
                    or assessor.get("identity_sha256") != identity
                    or assessor.get("task") != row.get("task")
                ):
                    raise Q36MTRScoreError("Q36 assessor content differs")
                rows[identity] = row
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q36MTRScoreError("Q36 assessor board is unreadable") from error
    observed = digest.hexdigest()
    if observed != expected_sha256 or len(rows) != TOTAL_ROWS:
        raise Q36MTRScoreError("Q36 assessor hash or cardinality differs")
    return rows, observed


def _identity_order(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        ("\n".join(str(row["identity_sha256"]) for row in rows) + "\n").encode()
    ).hexdigest()


def _load_selections(path: Path) -> dict[str, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRScoreError("Q36 selections are absent or symbolic")
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        identity = row.get("identity_sha256")
        selected = row.get("selected_index")
        margin = row.get("margin")
        if (
            set(row)
            != {
                "schema",
                "identity_sha256",
                "task",
                "selected_index",
                "selected_lineage",
                "order_consistent",
                "margin",
            }
            or row.get("schema") != SELECTION_SCHEMA
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in result
            or row.get("task") not in TASKS
            or selected not in (0, 1)
            or row.get("selected_lineage") != ("revision", "unchanged")[selected]
            or not isinstance(row.get("order_consistent"), bool)
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
        ):
            raise Q36MTRScoreError("Q36 selection content differs")
        result[identity] = row
    if len(result) != TOTAL_ROWS:
        raise Q36MTRScoreError("Q36 selection coverage differs")
    return result


def _input_hashes(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "development_data_sha256": sha256_file(args.development_data),
        "data_report_sha256": sha256_file(args.data_report),
        "assessor_receipt_sha256": sha256_file(args.assessor_receipt),
        "arm_report_sha256s": {
            arm: sha256_file(Path(getattr(args, f"{arm}_report"))) for arm in ARMS
        },
        "arm_candidate_sha256s": {
            arm: sha256_file(Path(getattr(args, f"{arm}_candidates"))) for arm in ARMS
        },
        "application_report_sha256": sha256_file(args.application_report),
        "selections_sha256": sha256_file(args.selections),
        "commit_training_report_sha256": sha256_file(args.commit_training_report),
        "precompute_custody_sha256": sha256_file(args.precompute_custody),
        "prescore_accounting_sha256": sha256_file(args.prescore_accounting),
        "environment_receipt_sha256": sha256_file(args.environment_receipt),
        "graph_contract_sha256": sha256_file(args.graph_contract),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRScoreError(f"refusing existing Q36 score artifact: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise Q36MTRScoreError("Q36 score artifact publication race") from error
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _consume(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return _atomic_json(path, payload)


def _publish_root(
    output: Path, outcomes: list[dict[str, Any]], report: dict[str, Any]
) -> None:
    if output.exists() or output.is_symlink():
        raise Q36MTRScoreError("Q36 score root exists")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        outcome_path = temporary / "outcomes.jsonl"
        digest = hashlib.sha256()
        with outcome_path.open("xb") as handle:
            for row in outcomes:
                encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
                handle.write(encoded)
                digest.update(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        final = {
            **report,
            "outcomes": str((output / "outcomes.jsonl").resolve()),
            "outcomes_sha256": digest.hexdigest(),
            "outcome_rows": len(outcomes),
        }
        with (temporary / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(final, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _metrics(
    sources: list[dict[str, Any]], correctness: dict[str, bool]
) -> dict[str, dict[str, int]]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for source in sources:
        identity = str(source["identity_sha256"])
        for domain in ("overall", str(source["task"])):
            buckets[domain]["total"] += 1
            buckets[domain]["correct"] += int(correctness[identity])
    return {domain: dict(counter) for domain, counter in sorted(buckets.items())}


def _score_impl(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise Q36MTRScoreError("Q36 score output already exists")
    sources = load_rows(args.development_data, "development")
    identity_order_sha256 = _identity_order(sources)
    source_by_id = {str(row["identity_sha256"]): row for row in sources}
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        values, _ = _load_arm(
            Path(getattr(args, f"{arm}_report")),
            Path(getattr(args, f"{arm}_candidates")),
            args.candidates_root,
            arm,
            "development",
        )
        if set(values) != set(source_by_id):
            raise Q36MTRScoreError(f"Q36 {arm} identity coverage differs")
        candidates[arm] = values
    selections = _load_selections(args.selections)
    if set(selections) != set(source_by_id):
        raise Q36MTRScoreError("Q36 selection/source coverage differs")
    application = json.loads(args.application_report.read_text(encoding="utf-8"))
    commit_training = json.loads(
        args.commit_training_report.read_text(encoding="utf-8")
    )
    application_truncated = application.get("prompt_truncated")
    application_malformed = application.get("malformed")
    application_consistent = application.get("order_consistent")
    if (
        application.get("schema") != APPLICATION_SCHEMA
        or application.get("status") != "complete"
        or application.get("model_revision") != MODEL_REVISION
        or application.get("rows") != TOTAL_ROWS
        or application.get("selections_sha256") != sha256_file(args.selections)
        or Path(str(application.get("selections", ""))).resolve()
        != args.selections.resolve()
        or isinstance(application_truncated, bool)
        or not isinstance(application_truncated, int)
        or not 0 <= application_truncated <= TOTAL_ROWS * 2
        or isinstance(application_malformed, bool)
        or not isinstance(application_malformed, int)
        or not 0 <= application_malformed <= TOTAL_ROWS
        or isinstance(application_consistent, bool)
        or not isinstance(application_consistent, int)
        or not 0 <= application_consistent <= TOTAL_ROWS
        or application_consistent
        != sum(int(row["order_consistent"]) for row in selections.values())
        or application.get("inference_fields")
        != ["question", "candidate_a", "candidate_b"]
        or application.get("correctness_or_task_label_visible") is not False
        or application.get("assessor_board_access_count") != 0
        or application.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
    ):
        raise Q36MTRScoreError("Q36 commit application differs")
    training_truncated = commit_training.get("training_prompt_truncated")
    calibration_truncated = commit_training.get(
        "calibration_development_prompt_truncated"
    )
    if (
        commit_training.get("schema") != COMMIT_REPORT_SCHEMA
        or commit_training.get("status") != "complete"
        or commit_training.get("model_revision") != MODEL_REVISION
        or commit_training.get("checkpoint_sha256")
        != application.get("commit_checkpoint_sha256")
        or Path(
            str(commit_training.get("development_application_report", ""))
        ).resolve()
        != args.application_report.resolve()
        or commit_training.get("development_selections_sha256")
        != sha256_file(args.selections)
        or commit_training.get("protected_adapter_unchanged") is not True
        or isinstance(training_truncated, bool)
        or not isinstance(training_truncated, int)
        or training_truncated < 0
        or isinstance(calibration_truncated, bool)
        or not isinstance(calibration_truncated, int)
        or calibration_truncated < 0
        or commit_training.get("sealed_access")
        != {"holdout": 0, "product": 0, "public": 0}
    ):
        raise Q36MTRScoreError("Q36 commit training report differs")
    assessor_receipt = json.loads(args.assessor_receipt.read_text(encoding="utf-8"))
    assessor_sha256 = assessor_receipt.get("board_sha256")
    if (
        assessor_receipt.get("schema") != CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA
        or assessor_receipt.get("status") != "complete"
        or assessor_receipt.get("rows") != TOTAL_ROWS
        or assessor_receipt.get("semantic_access") != "final_score_only"
        or not isinstance(assessor_sha256, str)
        or len(assessor_sha256) != 64
    ):
        raise Q36MTRScoreError("Q36 assessor receipt differs")
    expected_hashes = _input_hashes(args)
    authorization = json.loads(args.score_authorization.read_text(encoding="utf-8"))
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("status") != "complete"
        or authorization.get("scoring_authorized") is not True
        or authorization.get("one_shot") is not True
        or authorization.get("model_revision") != MODEL_REVISION
        or authorization.get("rows") != TOTAL_ROWS
        or authorization.get("identity_order_sha256") != identity_order_sha256
        or authorization.get("assessor_board_sha256") != assessor_sha256
        or authorization.get("score_output_root") != str(args.output.resolve())
        or authorization.get("assessor_board_access_count_before") != 0
        or authorization.get("input_hashes") != expected_hashes
        or authorization.get("code_sandbox_config_sha256") != SANDBOX_CONFIG_SHA256
        or authorization.get("code_sandbox_binary_sha256") != BWRAP_SHA256
        or authorization.get("sealed_access")
        != {"holdout": 0, "product": 0, "public": 0}
    ):
        raise Q36MTRScoreError("Q36 score authorization differs")
    run_id = authorization.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise Q36MTRScoreError("Q36 run identity differs")
    sandbox_payload = qualify_allocation()
    sandbox_receipt_sha256 = sandbox_atomic_json(
        args.sandbox_receipt_output, sandbox_payload
    )
    authorization_sha256 = sha256_file(args.score_authorization)
    consumption_path = args.output.with_name(
        f"{args.output.name}.score-authorization-consumed.json"
    )
    consumption_sha256 = _consume(
        consumption_path,
        {
            "schema": CONSUMPTION_SCHEMA,
            "status": "consumed",
            "run_id": run_id,
            "authorization_sha256": authorization_sha256,
            "score_output_root": str(args.output.resolve()),
            "input_hashes": expected_hashes,
        },
    )
    assessors, observed_board_sha256 = _load_assessors_once(
        args.assessor_board, assessor_sha256
    )
    if set(assessors) != set(source_by_id):
        raise Q36MTRScoreError("Q36 assessor/source identity coverage differs")
    setup_receipts = qualify_mbpp_assessor_setups(
        [assessors[str(row["identity_sha256"])]["assessor"] for row in sources]
    )
    correctness: dict[str, dict[str, bool]] = {arm: {} for arm in ARMS}
    malformed = Counter()
    policy_rejections = Counter()
    outcomes: list[dict[str, Any]] = []
    for source in sources:
        identity = str(source["identity_sha256"])
        assessor = assessors[identity]
        if assessor.get("task") != source.get("task"):
            raise Q36MTRScoreError("Q36 assessor task binding differs")
        local: dict[str, bool] = {}
        local_policy: dict[str, bool] = {}
        for arm in ARMS:
            completion = str(candidates[arm][identity]["completion"])
            result = score_completion(assessor["assessor"], completion)
            local[arm] = bool(result["correct"])
            correctness[arm][identity] = local[arm]
            malformed[arm] += int(not completion.strip())
            local_policy[arm] = bool(result.get("capability_policy_rejected", False))
            policy_rejections[arm] += int(local_policy[arm])
        selected = int(selections[identity]["selected_index"])
        commit_correct = (local["revision"], local["unchanged"])[selected]
        outcomes.append(
            {
                "schema": OUTCOME_SCHEMA,
                "identity_sha256": identity,
                "task": source["task"],
                "correct": {**local, "learned_commit": commit_correct},
                "capability_policy_rejected": local_policy,
                "selected_index": selected,
                "selected_lineage": selections[identity]["selected_lineage"],
                "order_consistent": selections[identity]["order_consistent"],
            }
        )
    commit_correctness = {
        row["identity_sha256"]: bool(row["correct"]["learned_commit"])
        for row in outcomes
    }
    metrics = {arm: _metrics(sources, correctness[arm]) for arm in ARMS}
    metrics["learned_commit"] = _metrics(sources, commit_correctness)
    malformed["learned_commit"] = sum(
        int(
            not candidates[("revision", "unchanged")[row["selected_index"]]][
                row["identity_sha256"]
            ]["completion"].strip()
        )
        for row in outcomes
    )
    policy_rejections["learned_commit"] = sum(
        int(
            row["capability_policy_rejected"][
                ("revision", "unchanged")[row["selected_index"]]
            ]
        )
        for row in outcomes
    )
    malformed_completions = {
        arm: sum(
            int(
                not candidates[arm][row["identity_sha256"]]["completion"].strip()
                or row["capability_policy_rejected"].get(arm, False)
            )
            for row in outcomes
        )
        for arm in ARMS
    }
    malformed_completions["learned_commit"] = sum(
        int(
            not candidates[("revision", "unchanged")[row["selected_index"]]][
                row["identity_sha256"]
            ]["completion"].strip()
            or row["capability_policy_rejected"].get(
                ("revision", "unchanged")[row["selected_index"]], False
            )
        )
        for row in outcomes
    )
    revision_correct = sum(correctness["revision"].values())
    unchanged_correct = sum(correctness["unchanged"].values())
    revision_retained = sum(
        correctness["revision"][row["identity_sha256"]]
        and row["correct"]["learned_commit"]
        for row in outcomes
    )
    unchanged_retained = sum(
        correctness["unchanged"][row["identity_sha256"]]
        and row["correct"]["learned_commit"]
        for row in outcomes
    )
    generation_truncation = {
        arm: sum(
            int(candidates[arm][identity]["max_token_exhausted"])
            for identity in source_by_id
        )
        for arm in ARMS
    }
    generation_truncation["learned_commit"] = sum(
        int(
            candidates[("revision", "unchanged")[row["selected_index"]]][
                row["identity_sha256"]
            ]["max_token_exhausted"]
        )
        for row in outcomes
    )
    report = {
        "schema": SCORE_SCHEMA,
        "status": "complete",
        "run_id": run_id,
        "model_revision": MODEL_REVISION,
        "rows": TOTAL_ROWS,
        "identity_order_sha256": identity_order_sha256,
        "metrics": metrics,
        "retention": {
            "revision_correct": {
                "retained": revision_retained,
                "total": revision_correct,
            },
            "unchanged_correct": {
                "retained": unchanged_retained,
                "total": unchanged_correct,
            },
        },
        "order_consistency": {
            "consistent": sum(int(row["order_consistent"]) for row in outcomes),
            "total": TOTAL_ROWS,
        },
        "publication_analysis": build_publication_analysis(outcomes),
        "empty_completion_counts": dict(malformed),
        "capability_policy_rejection_counts": dict(policy_rejections),
        "malformed_completion_counts": malformed_completions,
        "generation_truncation_counts": generation_truncation,
        "commit_prompt_truncated": application["prompt_truncated"],
        "commit_training_prompt_truncated": training_truncated + calibration_truncated,
        "commit_malformed": application["malformed"],
        "assessor_board_sha256": observed_board_sha256,
        "assessor_semantic_reads": 1,
        "assessor_rows_read": TOTAL_ROWS,
        "score_authorization_sha256": authorization_sha256,
        "score_consumption": str(consumption_path.resolve()),
        "score_consumption_sha256": consumption_sha256,
        "score_consumption_state": "consumed",
        "sandbox_receipt_sha256": sandbox_receipt_sha256,
        "sandbox_probe_sha256": sandbox_payload["probe_sha256"],
        "mbpp_setup_receipts": setup_receipts,
        "mbpp_setup_receipts_sha256": mbpp_allocation_setup_receipts_sha256(
            setup_receipts
        ),
        "input_hashes": expected_hashes,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _publish_root(args.output, outcomes, report)
    return report


def _terminal_failure_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.terminal-failure.json")


def _preserve_post_consumption_failure(
    args: argparse.Namespace, error: BaseException
) -> None:
    """Publish conservative terminal evidence only for this consumed authorization."""

    consumption_path = args.output.with_name(
        f"{args.output.name}.score-authorization-consumed.json"
    )
    failure_path = _terminal_failure_path(args.output)
    if (
        args.output.exists()
        or args.output.is_symlink()
        or failure_path.exists()
        or failure_path.is_symlink()
        or consumption_path.is_symlink()
        or not consumption_path.is_file()
        or args.score_authorization.is_symlink()
        or not args.score_authorization.is_file()
    ):
        return
    try:
        consumption = json.loads(consumption_path.read_text(encoding="utf-8"))
        authorization_sha256 = sha256_file(args.score_authorization)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if (
        consumption.get("schema") != CONSUMPTION_SCHEMA
        or consumption.get("status") != "consumed"
        or consumption.get("authorization_sha256") != authorization_sha256
        or consumption.get("score_output_root") != str(args.output.resolve())
    ):
        return
    _atomic_json(
        failure_path,
        {
            "schema": TERMINAL_FAILURE_SCHEMA,
            "status": "terminal_infrastructure_failure",
            "run_id": consumption.get("run_id"),
            "score_output_root": str(args.output.resolve()),
            "score_authorization_sha256": authorization_sha256,
            "score_consumption": str(consumption_path.resolve()),
            "score_consumption_sha256": sha256_file(consumption_path),
            "score_consumption_state": "consumed",
            "failure_stage": "after_one_shot_consumption_before_atomic_score_publication",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "assessor_semantic_read_state": "zero_or_partial_unknown",
            "retry_authorized": False,
            "successor_authorized": False,
            "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        },
    )


def score(args: argparse.Namespace) -> dict[str, Any]:
    try:
        return _score_impl(args)
    except BaseException as error:
        _preserve_post_consumption_failure(args, error)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--assessor-board", type=Path, required=True)
    parser.add_argument("--assessor-receipt", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    for arm in ARMS:
        parser.add_argument(
            f"--{arm.replace('_', '-')}-report", type=Path, required=True
        )
        parser.add_argument(
            f"--{arm.replace('_', '-')}-candidates", type=Path, required=True
        )
    parser.add_argument("--application-report", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--commit-training-report", type=Path, required=True)
    parser.add_argument("--precompute-custody", type=Path, required=True)
    parser.add_argument("--prescore-accounting", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--graph-contract", type=Path, required=True)
    parser.add_argument("--score-authorization", type=Path, required=True)
    parser.add_argument("--sandbox-receipt-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = score(parse_args())
    print(json.dumps({"status": report["status"], "rows": report["rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
