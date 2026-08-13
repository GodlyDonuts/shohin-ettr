#!/usr/bin/env python3
"""Reduce the five Q36-MTR arms to its single terminal development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from q36_mtr_contract import ARMS, MODEL_REVISION, TOTAL_ROWS, validate_graph

ARM_SCHEMA = "shohin-q36-mtr-arm-report-v1"
CUSTODY_SCHEMA = "shohin-q36-mtr-final-custody-v1"
OUTPUT_SCHEMA = "shohin-q36-mtr-final-comparison-v1"
DOMAINS = ("math500", "bbh_logic", "mbpp")


class Q36MTRComparisonError(RuntimeError):
    """Inputs cannot support the frozen Q36-MTR comparison."""


def _hex_digest(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonnegative_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _finite_nonnegative(value: object) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        return None
    return float(value)


def _load(path: Path, schema: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q36MTRComparisonError(f"unreadable Q36-MTR {label}: {path}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("status") != "complete"
    ):
        raise Q36MTRComparisonError(f"incomplete Q36-MTR {label}: {path}")
    return value, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _load_graph(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError("graph is not an object")
        validate_graph(value)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RuntimeError,
        TypeError,
    ) as error:
        raise Q36MTRComparisonError(
            f"invalid Q36-MTR graph contract: {path}"
        ) from error
    return value, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _metric(report: dict[str, Any], domain: str) -> dict[str, int]:
    value = report.get("metrics", {}).get(domain)
    if not isinstance(value, dict):
        raise Q36MTRComparisonError(f"missing Q36-MTR metric: {domain}")
    correct = _nonnegative_integer(value.get("correct"))
    total = _nonnegative_integer(value.get("total"))
    if correct is None or total is None or correct > total:
        raise Q36MTRComparisonError(f"malformed Q36-MTR metric: {domain}")
    return {"correct": correct, "total": total}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRComparisonError(f"refusing existing Q36-MTR result: {path}")
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
        raise Q36MTRComparisonError(
            f"refusing existing Q36-MTR result: {path}"
        ) from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _at_least_95_percent(retained: int | None, total: int | None) -> bool:
    return (
        retained is not None
        and total is not None
        and total > 0
        and retained <= total
        and retained * 100 >= total * 95
    )


def compare(args: argparse.Namespace) -> dict[str, Any]:
    arm_paths = {arm: Path(getattr(args, f"{arm}_report")) for arm in ARMS}
    custody_path = Path(args.final_custody)
    output = Path(args.output)
    graph, graph_receipt = _load_graph(Path(args.graph_contract))
    arms: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for arm, path in arm_paths.items():
        report, receipt = _load(path, ARM_SCHEMA, arm)
        if report.get("arm") != arm:
            raise Q36MTRComparisonError(f"Q36-MTR arm label differs: {arm}")
        arms[arm] = report
        inputs[arm] = receipt
    custody, custody_receipt = _load(custody_path, CUSTODY_SCHEMA, "custody")
    inputs["graph_contract"] = graph_receipt
    inputs["final_custody"] = custody_receipt

    expected_arm_hashes = {arm: inputs[arm]["sha256"] for arm in ARMS}
    if custody.get("arm_report_sha256s") != expected_arm_hashes:
        raise Q36MTRComparisonError("Q36-MTR arm hashes differ from custody")
    if custody.get("graph_contract_sha256") != graph_receipt["sha256"] or custody.get(
        "source_commit"
    ) != graph.get("source_commit"):
        raise Q36MTRComparisonError("Q36-MTR graph binding differs from custody")

    run_ids = {report.get("run_id") for report in arms.values()}
    run_ids.add(custody.get("run_id"))
    if len(run_ids) != 1 or not all(
        isinstance(value, str) and value for value in run_ids
    ):
        raise Q36MTRComparisonError("Q36-MTR run identity differs")
    identity_hashes = {report.get("identity_order_sha256") for report in arms.values()}
    identity_hashes.add(custody.get("identity_order_sha256"))
    if len(identity_hashes) != 1 or not all(
        _hex_digest(value) for value in identity_hashes
    ):
        raise Q36MTRComparisonError("Q36-MTR identity order differs")
    data_hashes = {report.get("data_sha256") for report in arms.values()}
    data_hashes.add(custody.get("data_sha256"))
    runtime_hashes = {report.get("runtime_sha256") for report in arms.values()}
    runtime_hashes.add(custody.get("runtime_sha256"))
    if len(data_hashes) != 1 or not all(_hex_digest(value) for value in data_hashes):
        raise Q36MTRComparisonError("Q36-MTR data binding differs")
    if len(runtime_hashes) != 1 or not all(
        _hex_digest(value) for value in runtime_hashes
    ):
        raise Q36MTRComparisonError("Q36-MTR runtime binding differs")
    if custody.get("model_revision") != MODEL_REVISION or any(
        report.get("model_revision") != MODEL_REVISION for report in arms.values()
    ):
        raise Q36MTRComparisonError("Q36-MTR model revision differs")
    precompute_hashes = {
        report.get("precompute_custody_sha256") for report in arms.values()
    }
    precompute_hashes.add(custody.get("precompute_custody_sha256"))
    if len(precompute_hashes) != 1 or not all(
        _hex_digest(value) for value in precompute_hashes
    ):
        raise Q36MTRComparisonError("Q36-MTR precompute custody binding differs")

    metrics: dict[str, dict[str, dict[str, int]]] = {}
    for arm, report in arms.items():
        if report.get("split") != "development":
            raise Q36MTRComparisonError(f"Q36-MTR split differs: {arm}")
        metrics[arm] = {
            domain: _metric(report, domain) for domain in ("overall", *DOMAINS)
        }
        domain_total = sum(metrics[arm][domain]["total"] for domain in DOMAINS)
        domain_correct = sum(metrics[arm][domain]["correct"] for domain in DOMAINS)
        if (
            _nonnegative_integer(report.get("full_row_count")) != TOTAL_ROWS
            or metrics[arm]["overall"]["total"] != TOTAL_ROWS
            or domain_total != TOTAL_ROWS
            or domain_correct != metrics[arm]["overall"]["correct"]
        ):
            raise Q36MTRComparisonError(f"Q36-MTR metric arithmetic differs: {arm}")

    scores = {arm: metrics[arm]["overall"]["correct"] for arm in ARMS}
    domain_scores = {
        arm: {domain: metrics[arm][domain]["correct"] for domain in DOMAINS}
        for arm in ARMS
    }
    domain_totals = {
        tuple(metrics[arm][domain]["total"] for domain in DOMAINS) for arm in ARMS
    }
    if len(domain_totals) != 1:
        raise Q36MTRComparisonError("Q36-MTR domain totals differ between arms")

    revision = scores["trained_revision"]
    unchanged = scores["unchanged"]
    self_refinement = scores["self_refinement"]
    draft_hidden = scores["draft_hidden"]
    commit = scores["learned_commit"]
    revision_deltas = {
        control: {
            domain: domain_scores["trained_revision"][domain]
            - domain_scores[control][domain]
            for domain in DOMAINS
        }
        for control in ("unchanged", "self_refinement", "draft_hidden")
    }
    commit_deltas = {
        control: {
            domain: domain_scores["learned_commit"][domain]
            - domain_scores[control][domain]
            for domain in DOMAINS
        }
        for control in ("trained_revision", "unchanged")
    }

    retention = arms["learned_commit"].get("retention")
    order = arms["learned_commit"].get("order_consistency")
    if not isinstance(retention, dict) or not isinstance(order, dict):
        raise Q36MTRComparisonError("Q36-MTR commit evidence is missing")
    revision_retention = retention.get("revision_correct")
    unchanged_retention = retention.get("unchanged_correct")
    if not isinstance(revision_retention, dict) or not isinstance(
        unchanged_retention, dict
    ):
        raise Q36MTRComparisonError("Q36-MTR retention evidence is missing")
    revision_retained = _nonnegative_integer(revision_retention.get("retained"))
    revision_retention_total = _nonnegative_integer(revision_retention.get("total"))
    unchanged_retained = _nonnegative_integer(unchanged_retention.get("retained"))
    unchanged_retention_total = _nonnegative_integer(unchanged_retention.get("total"))

    checks: dict[str, bool] = {
        "unchanged_at_least_387": unchanged >= 387,
        "unchanged_all_domains_nonzero": all(
            domain_scores["unchanged"][domain] > 0 for domain in DOMAINS
        ),
        "revision_at_least_65_over_unchanged": revision >= unchanged + 65,
        "revision_at_least_39_over_self_refinement": revision >= self_refinement + 39,
        "revision_at_least_39_over_draft_hidden": revision >= draft_hidden + 39,
        "revision_domain_deltas_vs_all_controls_nonnegative": all(
            delta >= 0
            for control in revision_deltas.values()
            for delta in control.values()
        ),
        "commit_at_least_13_over_revision": commit >= revision + 13,
        "commit_domain_deltas_vs_revision_and_unchanged_nonnegative": all(
            delta >= 0
            for control in commit_deltas.values()
            for delta in control.values()
        ),
        "commit_revision_retention_denominator_matches": revision_retention_total
        == revision,
        "commit_revision_retention_at_least_95_percent": _at_least_95_percent(
            revision_retained, revision_retention_total
        ),
        "commit_unchanged_retention_denominator_matches": unchanged_retention_total
        == unchanged,
        "commit_unchanged_retention_at_least_95_percent": _at_least_95_percent(
            unchanged_retained, unchanged_retention_total
        ),
        "commit_exact_ab_order_consistency": _nonnegative_integer(
            order.get("consistent")
        )
        == TOTAL_ROWS
        and _nonnegative_integer(order.get("total")) == TOTAL_ROWS,
    }
    for arm, report in arms.items():
        checks[f"{arm}_zero_truncation"] = (
            _nonnegative_integer(report.get("truncation_count")) == 0
        )
        checks[f"{arm}_zero_malformed"] = (
            _nonnegative_integer(report.get("malformed_count")) == 0
        )
        checks[f"{arm}_one_output_per_identity"] = (
            _nonnegative_integer(report.get("candidate_count")) == TOTAL_ROWS
        )

    custody_checks = {
        "custody_verified": custody.get("custody_verified") is True,
        "source_disjoint": custody.get("source_disjoint") is True,
        "model_manifest_verified": custody.get("model_manifest_verified") is True
        and _hex_digest(custody.get("model_manifest_sha256")),
        "runtime_manifest_verified": custody.get("runtime_manifest_verified") is True
        and _hex_digest(custody.get("runtime_manifest_sha256"))
        and custody.get("runtime_source_commit") == graph.get("source_commit"),
        "checkpoint_manifests_verified": custody.get("checkpoint_manifests_verified")
        is True
        and isinstance(custody.get("checkpoint_manifest_sha256s"), dict)
        and set(custody["checkpoint_manifest_sha256s"])
        == {"owner", "trained_revision", "draft_hidden", "learned_commit"}
        and all(
            _hex_digest(value)
            for value in custody["checkpoint_manifest_sha256s"].values()
        ),
        "environment_verified": custody.get("environment_verified") is True
        and _hex_digest(custody.get("environment_receipt_sha256")),
        "sandbox_verified": custody.get("sandbox_verified") is True
        and _hex_digest(custody.get("sandbox_receipt_sha256")),
        "scheduler_accounting_verified": custody.get("scheduler_accounting_verified")
        is True
        and _hex_digest(custody.get("scheduler_accounting_sha256")),
        "one_assessor_open_verified": custody.get("one_assessor_open_verified") is True
        and _nonnegative_integer(custody.get("assessor_semantic_reads")) == 1,
        "protected_access_zero": all(
            _nonnegative_integer(custody.get(field)) == 0
            for field in (
                "public_access_count",
                "holdout_access_count",
                "product_access_count",
            )
        ),
        "retry_requeue_duplicate_orphan_zero": all(
            _nonnegative_integer(custody.get(field)) == 0
            for field in (
                "retry_count",
                "requeue_count",
                "duplicate_shard_count",
                "orphaned_job_count",
            )
        ),
        "no_successor": custody.get("successor_authorized") is False
        and custody.get("successor_submitted") is False,
        "score_consumed_once": custody.get("score_consumption_state") == "consumed"
        and _hex_digest(custody.get("score_consumption_sha256")),
        "graph_contract_bound": custody.get("graph_contract_sha256")
        == graph_receipt["sha256"],
        "source_commit_bound": custody.get("source_commit")
        == graph.get("source_commit"),
        "exact_h100_request_count": _nonnegative_integer(
            custody.get("h100_request_count")
        )
        == 61
        and _nonnegative_integer(custody.get("completed_h100_allocation_count")) == 61,
        "charged_gpu_seconds_recorded": _finite_nonnegative(
            custody.get("charged_gpu_seconds")
        )
        not in (None, 0.0),
        "evidence_mirror_verified": custody.get("evidence_mirror_verified") is True
        and _hex_digest(custody.get("evidence_mirror_manifest_sha256")),
    }
    checks.update({f"custody_{name}": value for name, value in custody_checks.items()})

    gates = {
        "capable_host": checks["unchanged_at_least_387"]
        and checks["unchanged_all_domains_nonzero"],
        "causal_revision": all(
            checks[name]
            for name in (
                "revision_at_least_65_over_unchanged",
                "revision_at_least_39_over_self_refinement",
                "revision_at_least_39_over_draft_hidden",
                "revision_domain_deltas_vs_all_controls_nonnegative",
            )
        ),
        "useful_learned_commit": checks["commit_at_least_13_over_revision"]
        and checks["commit_domain_deltas_vs_revision_and_unchanged_nonnegative"],
        "conservative_retention": all(
            checks[name]
            for name in (
                "commit_revision_retention_denominator_matches",
                "commit_revision_retention_at_least_95_percent",
                "commit_unchanged_retention_denominator_matches",
                "commit_unchanged_retention_at_least_95_percent",
            )
        ),
        "complete_outputs": checks["commit_exact_ab_order_consistency"]
        and all(
            checks[f"{arm}_{suffix}"]
            for arm in ARMS
            for suffix in (
                "zero_truncation",
                "zero_malformed",
                "one_output_per_identity",
            )
        ),
        "complete_custody": all(custody_checks.values()),
    }
    gate_pass = all(gates.values())
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "run_id": next(iter(run_ids)),
        "model_revision": MODEL_REVISION,
        "formal_result": "PASS" if gate_pass else "FAIL",
        "gate_pass": gate_pass,
        "scores": scores,
        "domain_scores": domain_scores,
        "margins": {
            "revision_minus_unchanged": revision - unchanged,
            "revision_minus_self_refinement": revision - self_refinement,
            "revision_minus_draft_hidden": revision - draft_hidden,
            "commit_minus_revision": commit - revision,
        },
        "domain_deltas": {
            "revision": revision_deltas,
            "commit": commit_deltas,
        },
        "retention": {
            "revision_correct": {
                "retained": revision_retained,
                "total": revision_retention_total,
            },
            "unchanged_correct": {
                "retained": unchanged_retained,
                "total": unchanged_retention_total,
            },
        },
        "checks": checks,
        "gates": gates,
        "inputs": inputs,
        "stop_after_gate": True,
        "automatic_retry_authorized": False,
        "automatic_confirmation_authorized": False,
        "automatic_successor_authorized": False,
        "holdout_access_authorized": False,
        "product_access_authorized": False,
        "next_action": "stop_and_preserve_evidence",
        "claim_boundary": "source_disjoint_development_mechanism_only",
    }
    _atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}-report", required=True)
    parser.add_argument("--final-custody", required=True)
    parser.add_argument("--graph-contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
