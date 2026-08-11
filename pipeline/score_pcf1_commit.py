#!/usr/bin/env python3
"""Open and score the frozen PCF1 confirmation board exactly once.

Every GPU-produced confirmation candidate and learned-commit selection is
label-free.  This CPU custodian validates all pre-score evidence and the
explicit score authorization before it performs the sole semantic read of
the assessor board.  Revision, unchanged, self-refinement, and selected
trajectory outcomes are then scored in one process and one write-once result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

from build_pcf1_commit_pairs import PCF1PairError, load_arm, sha256_file
from build_pcf1_confirmation_pairs import REPORT_SCHEMA as PAIR_REPORT_SCHEMA
from build_pcf1_data import (
    CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA,
    CONFIRMATION_ASSESSOR_SCHEMA,
)
from hf_pcf1_apply_commit import (
    REPORT_SCHEMA as APPLICATION_REPORT_SCHEMA,
    SELECTION_SCHEMA,
)
from hf_pcf1_evaluate import DATA_REPORT_SCHEMA, load_rows
from hf_pcf1_train_commit import REPORT_SCHEMA as TRAINING_REPORT_SCHEMA
from pcf1_code_sandbox import (
    BWRAP_SHA256,
    SANDBOX_CONFIG_SHA256,
    atomic_json as sandbox_atomic_json,
    mbpp_allocation_setup_receipts_sha256,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
    score_completion,
)
from pcf1_environment import validate_environment_receipt

REPORT_SCHEMA = "shohin-pcf1-commit-result-v1"
OUTCOME_SCHEMA = "shohin-pcf1-confirmation-outcome-v1"
AUTHORIZATION_SCHEMA = "shohin-pcf1-score-authorization-v1"
CONSUMPTION_SCHEMA = "shohin-pcf1-score-consumption-v1"
TERMINAL_FAILURE_SCHEMA = "shohin-pcf1-score-terminal-failure-v1"
TASKS = ("math500", "bbh_logic", "mbpp")
ARMS = ("revision", "unchanged", "self_refinement")
TOTAL_ROWS = 1289
PINNED_MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
SEALED_ZERO = {"holdout": 0, "product": 0, "public": 0}


class PCF1ScoreError(RuntimeError):
    """The one-shot confirmation authorization, inputs, or outcomes differ."""


def reject_protected_path(path: Path) -> None:
    rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
    if any(word in rendered for word in ("holdout", "product", "public")):
        raise PCF1ScoreError(f"protected path supplied to PCF1 scoring: {path}")


def load_json(path: Path, schema: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PCF1ScoreError(f"unreadable PCF1 {label}: {path}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("status") != "complete"
    ):
        raise PCF1ScoreError(f"incomplete PCF1 {label}: {path}")
    return value


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def identity_order_sha256(sources: list[dict[str, Any]]) -> str:
    identities = sorted(str(row["identity_sha256"]) for row in sources)
    return hashlib.sha256(("\n".join(identities) + "\n").encode()).hexdigest()


def load_selections(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PCF1ScoreError(f"unreadable PCF1 selections: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PCF1ScoreError(
                f"malformed PCF1 selection at line {line_number}"
            ) from error
        identity = row.get("identity_sha256")
        selected_index = row.get("selected_index")
        if (
            row.get("schema") != SELECTION_SCHEMA
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in rows
            or row.get("task") not in TASKS
            or isinstance(selected_index, bool)
            or selected_index not in (0, 1)
            or row.get("selected_lineage") not in ("revision", "unchanged")
            or row.get("selected_lineage") != ("revision", "unchanged")[selected_index]
            or not isinstance(row.get("order_consistent"), bool)
            or isinstance(row.get("margin"), bool)
            or not isinstance(row.get("margin"), (int, float))
            or not math.isfinite(float(row["margin"]))
        ):
            raise PCF1ScoreError("PCF1 selection content differs")
        rows[identity] = row
    if len(rows) != TOTAL_ROWS:
        raise PCF1ScoreError("PCF1 selection cardinality differs")
    return rows


def validate_confirmation_pairs(
    path: Path,
    sources: list[dict[str, Any]],
    candidates: dict[str, dict[str, dict[str, Any]]],
) -> None:
    by_identity: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PCF1ScoreError("unreadable PCF1 confirmation pairs") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PCF1ScoreError(
                f"malformed PCF1 confirmation pair {line_number}"
            ) from error
        identity = row.get("identity_sha256")
        pair_candidates = row.get("candidates")
        if (
            row.get("schema") != "shohin-pcf1-confirmation-pair-v1"
            or row.get("split") != "confirmation"
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in by_identity
            or row.get("task") not in TASKS
            or not isinstance(row.get("question"), str)
            or not isinstance(pair_candidates, list)
            or len(pair_candidates) != 2
            or [candidate.get("lineage") for candidate in pair_candidates]
            != ["revision", "unchanged"]
            or any(
                set(candidate) != {"lineage", "completion"}
                or not isinstance(candidate.get("completion"), str)
                for candidate in pair_candidates
            )
        ):
            raise PCF1ScoreError("PCF1 confirmation pair content differs")
        by_identity[identity] = row
    source_by_identity = {str(row["identity_sha256"]): row for row in sources}
    if set(by_identity) != set(source_by_identity):
        raise PCF1ScoreError("PCF1 confirmation pair coverage differs")
    for identity, source in source_by_identity.items():
        row = by_identity[identity]
        expected = [
            candidates[arm][identity]["completion"] for arm in ("revision", "unchanged")
        ]
        if (
            row.get("task") != source.get("task")
            or row.get("question") != source.get("question")
            or [candidate["completion"] for candidate in row["candidates"]] != expected
        ):
            raise PCF1ScoreError("PCF1 confirmation pair/arm binding differs")


def load_assessors_once(
    path: Path,
    expected_sha256: str,
    progress: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Perform the sole semantic assessor-board read after authorization."""

    rows: dict[str, dict[str, Any]] = {}
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                hasher.update(raw_line)
                if progress is not None:
                    progress["assessor_bytes_read"] = int(
                        progress.get("assessor_bytes_read", 0)
                    ) + len(raw_line)
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise PCF1ScoreError(
                        f"malformed PCF1 assessor row {line_number}"
                    ) from error
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise PCF1ScoreError(
                        f"malformed PCF1 assessor row {line_number}"
                    ) from error
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
                    raise PCF1ScoreError("PCF1 confirmation assessor content differs")
                rows[identity] = row
                if progress is not None:
                    progress["assessor_rows_read"] = len(rows)
    except OSError as error:
        raise PCF1ScoreError("unreadable PCF1 confirmation assessor board") from error
    digest = hasher.hexdigest()
    if digest != expected_sha256:
        raise PCF1ScoreError("PCF1 confirmation assessor hash differs")
    if len(rows) != TOTAL_ROWS:
        raise PCF1ScoreError("PCF1 confirmation assessor cardinality differs")
    return rows, digest


def arm_metrics(
    sources: list[dict[str, Any]], correctness: dict[str, bool]
) -> dict[str, dict[str, int]]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for source in sources:
        identity = str(source["identity_sha256"])
        if identity not in correctness:
            raise PCF1ScoreError("PCF1 scored arm coverage differs")
        for domain in ("overall", str(source["task"])):
            buckets[domain]["total"] += 1
            buckets[domain]["generated_correct"] += int(correctness[identity])
    if set(buckets) != {"overall", *TASKS}:
        raise PCF1ScoreError("PCF1 scored arm domains differ")
    return {domain: dict(counter) for domain, counter in sorted(buckets.items())}


def commit_metrics(
    sources: list[dict[str, Any]],
    correctness: dict[str, dict[str, bool]],
    selections: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for source in sources:
        identity = str(source["identity_sha256"])
        selection = selections[identity]
        selected_index = int(selection["selected_index"])
        expected_lineage = ("revision", "unchanged")[selected_index]
        if (
            selection.get("task") != source.get("task")
            or selection.get("selected_lineage") != expected_lineage
        ):
            raise PCF1ScoreError("PCF1 selected lineage/task binding differs")
        revision_correct = correctness["revision"][identity]
        unchanged_correct = correctness["unchanged"][identity]
        selected_correct = (revision_correct, unchanged_correct)[selected_index]
        for domain in ("overall", str(source["task"])):
            bucket = buckets[domain]
            bucket["total"] += 1
            bucket["revision_correct"] += int(revision_correct)
            bucket["unchanged_correct"] += int(unchanged_correct)
            bucket["selected_correct"] += int(selected_correct)
            bucket["oracle_correct"] += int(revision_correct or unchanged_correct)
            bucket["unchanged_commits"] += int(selected_index == 1)
            bucket["order_consistent"] += int(selection["order_consistent"])
            bucket["revision_correct_retained"] += int(
                revision_correct and selected_correct
            )
            bucket["unchanged_correct_retained"] += int(
                unchanged_correct and selected_correct
            )
    result: dict[str, dict[str, Any]] = {}
    for domain in ("overall", *TASKS):
        bucket = buckets[domain]
        total = bucket["total"]
        result[domain] = {
            **dict(bucket),
            "selected_accuracy": bucket["selected_correct"] / total,
            "order_consistency": bucket["order_consistent"] / total,
            "revision_correct_retention": (
                bucket["revision_correct_retained"] / bucket["revision_correct"]
                if bucket["revision_correct"]
                else None
            ),
            "unchanged_correct_retention": (
                bucket["unchanged_correct_retained"] / bucket["unchanged_correct"]
                if bucket["unchanged_correct"]
                else None
            ),
        }
    return result


def publish_score_root(
    output_root: Path,
    outcomes: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    """Atomically publish outcomes and the terminal report as one directory."""

    if output_root.exists() or output_root.is_symlink():
        raise PCF1ScoreError(f"refusing existing PCF1 result: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.with_name(f".{output_root.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise PCF1ScoreError(f"refusing existing PCF1 temporary result: {temporary}")
    temporary.mkdir()
    try:
        outcomes_path = temporary / "outcomes.jsonl"
        digest = hashlib.sha256()
        with outcomes_path.open("xb") as handle:
            for row in outcomes:
                encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
                handle.write(encoded)
                digest.update(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        payload["outcomes"] = str((output_root / "outcomes.jsonl").resolve())
        payload["outcomes_sha256"] = digest.hexdigest()
        payload["outcome_rows"] = len(outcomes)
        with (temporary / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temporary, output_root)
        parent_fd = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if temporary.exists() and temporary.is_dir():
            shutil.rmtree(temporary)
        raise


def score_consumption_path(output_root: Path) -> Path:
    """Derive the one immutable authorization-consumption marker."""

    return output_root.with_name(
        f"{output_root.name}.score-authorization-consumed.json"
    )


def score_sandbox_probe_path(output_root: Path) -> Path:
    """Derive the only permitted final-score allocation probe path."""

    return output_root.with_name(f"{output_root.name}.sandbox-probe.json")


def score_terminal_failure_path(output_root: Path) -> Path:
    """Derive durable terminal evidence for a post-claim infrastructure failure."""

    return output_root.with_name(f"{output_root.name}.terminal-failure.json")


def consume_score_authorization(path: Path, payload: dict[str, Any]) -> str:
    """Publish a durable, write-once claim before the assessor board is opened."""

    if path.exists() or path.is_symlink():
        raise PCF1ScoreError("PCF1 score authorization is already consumed")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise PCF1ScoreError(
                "PCF1 score authorization is already consumed"
            ) from error
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def publish_terminal_failure(
    path: Path,
    args: argparse.Namespace,
    state: dict[str, Any],
    error: BaseException,
) -> str:
    """Publish one write-once failure receipt; a claimed score can never retry."""

    payload = {
        "schema": TERMINAL_FAILURE_SCHEMA,
        "status": "infrastructure_failure",
        "run_id": state.get("run_id"),
        "score_output_root": str(args.output_root.resolve()),
        "score_authorization_sha256": state.get("authorization_sha256"),
        "score_consumption": str(score_consumption_path(args.output_root).resolve()),
        "score_consumption_sha256": state.get("consumption_sha256"),
        "score_consumption_state": "consumed",
        "sandbox_probe_sha256": state.get("sandbox_probe_sha256"),
        "sandbox_probe_result_sha256": state.get("sandbox_probe_result_sha256"),
        "sandbox_receipt_sha256": state.get("sandbox_receipt_sha256"),
        "failure_phase": state.get("phase", "unknown"),
        "assessor_bytes_read": int(state.get("assessor_bytes_read", 0)),
        "assessor_rows_read": int(state.get("assessor_rows_read", 0)),
        "assessment_calls_started": int(state.get("assessment_calls_started", 0)),
        "sandbox_calls_started": int(state.get("sandbox_calls_started", 0)),
        "exception_class": type(error).__name__,
        "exception_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
        "retry_authorized": False,
        "retry_count": 0,
        "successor_authorized": False,
        "successor_submitted": False,
    }
    if any(
        value is None
        for value in (
            payload["run_id"],
            payload["score_authorization_sha256"],
            payload["score_consumption_sha256"],
            payload["sandbox_probe_sha256"],
            payload["sandbox_probe_result_sha256"],
            payload["sandbox_receipt_sha256"],
        )
    ):
        raise PCF1ScoreError("PCF1 post-claim failure state is incomplete") from error
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() or path.is_symlink():
        raise PCF1ScoreError("PCF1 terminal failure receipt already exists") from error
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as publication_error:
            raise PCF1ScoreError(
                "PCF1 terminal failure publication race"
            ) from publication_error
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _authorization_hashes(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "candidates_root": str(args.candidates_root.resolve(strict=True)),
        "confirmation_data_sha256": sha256_file(args.confirmation_data),
        "confirmation_assessor_receipt_sha256": sha256_file(
            args.confirmation_assessor_receipt
        ),
        "data_report_sha256": sha256_file(args.data_report),
        "arm_report_sha256s": {
            "revision": sha256_file(args.revision_report),
            "unchanged": sha256_file(args.unchanged_report),
            "self_refinement": sha256_file(args.self_refinement_report),
        },
        "arm_candidates_sha256s": {
            "revision": sha256_file(args.revision_candidates),
            "unchanged": sha256_file(args.unchanged_candidates),
            "self_refinement": sha256_file(args.self_refinement_candidates),
        },
        "confirmation_pairs_sha256": sha256_file(args.confirmation_pairs),
        "pair_report_sha256": sha256_file(args.confirmation_pairs_report),
        "commit_application_report_sha256": sha256_file(args.application_report),
        "selections_sha256": sha256_file(args.selections),
        "commit_training_report_sha256": sha256_file(args.training_report),
        "mechanics_report_sha256": sha256_file(args.mechanics_report),
        "data_custody_sha256": sha256_file(args.data_custody),
        "model_custody_sha256": sha256_file(args.model_custody),
        "runtime_custody_sha256": sha256_file(args.runtime_custody),
        "prescore_dispatch_receipt_sha256": sha256_file(args.prescore_dispatch_receipt),
        "prescore_accounting_receipt_sha256": sha256_file(
            args.prescore_accounting_receipt
        ),
        "environment_receipt_sha256": args.environment_receipt_sha256,
    }


def validate_authorization(
    authorization: dict[str, Any],
    expected_hashes: dict[str, Any],
    assessor_sha256: str,
    identity_sha256: str,
    output_root: Path,
    environment_tree_sha256: str,
) -> str:
    if (
        authorization.get("scoring_authorized") is not True
        or authorization.get("one_shot") is not True
        or authorization.get("rows") != TOTAL_ROWS
        or authorization.get("identity_order_sha256") != identity_sha256
        or authorization.get("confirmation_assessors_sha256") != assessor_sha256
        or authorization.get("score_output_root") != str(output_root.resolve())
        or authorization.get("assessor_board_access_count_before") != 0
        or authorization.get("sealed_access") != SEALED_ZERO
        or authorization.get("environment_tree_sha256") != environment_tree_sha256
        or authorization.get("code_sandbox_config_sha256") != SANDBOX_CONFIG_SHA256
        or authorization.get("code_sandbox_binary_sha256") != BWRAP_SHA256
        or not is_sha256(authorization.get("code_sandbox_probe_sha256"))
        or not is_sha256(authorization.get("code_sandbox_probe_result_sha256"))
        or not is_sha256(authorization.get("code_sandbox_receipt_sha256"))
        or any(
            authorization.get(key) != value for key, value in expected_hashes.items()
        )
    ):
        raise PCF1ScoreError("PCF1 score authorization binding differs")
    run_id = authorization.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise PCF1ScoreError("PCF1 score authorization run id differs")
    return run_id


def _score(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    environment = validate_environment_receipt(
        args.environment_receipt,
        args.environment_receipt_sha256,
        "pipeline/score_pcf1_commit.py",
    )
    explicit_paths = (
        args.confirmation_data,
        args.confirmation_assessors,
        args.confirmation_assessor_receipt,
        args.data_report,
        args.revision_report,
        args.revision_candidates,
        args.unchanged_report,
        args.unchanged_candidates,
        args.self_refinement_report,
        args.self_refinement_candidates,
        args.candidates_root,
        args.confirmation_pairs,
        args.confirmation_pairs_report,
        args.selections,
        args.application_report,
        args.training_report,
        args.mechanics_report,
        args.data_custody,
        args.model_custody,
        args.runtime_custody,
        args.prescore_dispatch_receipt,
        args.prescore_accounting_receipt,
        args.prescore_authorization,
        args.environment_receipt,
        args.sandbox_probe_output,
        args.output_root,
    )
    for path in explicit_paths:
        reject_protected_path(path)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise PCF1ScoreError(f"refusing existing PCF1 result: {args.output_root}")
    claim_path = score_consumption_path(args.output_root)
    if claim_path.exists() or claim_path.is_symlink():
        raise PCF1ScoreError("PCF1 score authorization is already consumed")
    for sibling in (
        score_terminal_failure_path(args.output_root),
        args.sandbox_probe_output,
    ):
        if sibling.exists() or sibling.is_symlink():
            raise PCF1ScoreError("PCF1 score sibling evidence already exists")

    data_report = load_json(args.data_report, DATA_REPORT_SCHEMA, "data report")
    expected_data = data_report.get("outputs", {}).get("confirmation")
    expected_assessors = data_report.get("outputs", {}).get("confirmation_assessors")
    expected_assessor_receipt = data_report.get("outputs", {}).get(
        "confirmation_assessor_receipt"
    )
    if (
        data_report.get("sealed_access") != SEALED_ZERO
        or data_report.get("confirmation_assessor_access")
        != {"semantic_reads": 0, "authorized_reader": "score_pcf1_commit.py"}
        or not isinstance(expected_data, dict)
        or Path(str(expected_data.get("path", ""))).resolve()
        != args.confirmation_data.resolve()
        or expected_data.get("sha256") != sha256_file(args.confirmation_data)
        or not isinstance(expected_assessors, dict)
        or expected_assessors.get("rows") != TOTAL_ROWS
        or expected_assessors.get("semantic_access") != "final_score_only"
        or not is_sha256(expected_assessors.get("sha256"))
        or not isinstance(expected_assessor_receipt, dict)
        or expected_assessor_receipt.get("rows") != 1
        or expected_assessor_receipt.get("sha256")
        != sha256_file(args.confirmation_assessor_receipt)
    ):
        raise PCF1ScoreError("PCF1 confirmation data receipt differs")
    assessor_receipt = load_json(
        args.confirmation_assessor_receipt,
        CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA,
        "confirmation assessor receipt",
    )
    if (
        assessor_receipt.get("board_sha256") != expected_assessors.get("sha256")
        or assessor_receipt.get("rows") != TOTAL_ROWS
        or assessor_receipt.get("semantic_access") != "final_score_only"
    ):
        raise PCF1ScoreError("PCF1 confirmation assessor receipt differs")
    sources = load_rows(args.confirmation_data, "confirmation")
    identity_sha256 = identity_order_sha256(sources)

    arm_paths = {
        "revision": (args.revision_report, args.revision_candidates),
        "unchanged": (args.unchanged_report, args.unchanged_candidates),
        "self_refinement": (
            args.self_refinement_report,
            args.self_refinement_candidates,
        ),
    }
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    arm_receipts: dict[str, dict[str, Any]] = {}
    for arm, (report_path, candidates_path) in arm_paths.items():
        try:
            loaded, receipt = load_arm(
                report_path,
                candidates_path,
                arm,
                "confirmation",
                candidates_root=args.candidates_root,
            )
        except PCF1PairError as error:
            raise PCF1ScoreError(f"PCF1 label-free {arm} arm differs") from error
        candidates[arm] = loaded
        arm_receipts[arm] = receipt
    validate_confirmation_pairs(args.confirmation_pairs, sources, candidates)

    pair_report = load_json(
        args.confirmation_pairs_report,
        PAIR_REPORT_SCHEMA,
        "confirmation pair report",
    )
    application = load_json(
        args.application_report,
        APPLICATION_REPORT_SCHEMA,
        "commit application report",
    )
    training = load_json(
        args.training_report,
        TRAINING_REPORT_SCHEMA,
        "commit training report",
    )
    confirmation_pairs_sha256 = sha256_file(args.confirmation_pairs)
    if (
        pair_report.get("rows") != TOTAL_ROWS
        or pair_report.get("labels_or_correctness_fields") != 0
        or pair_report.get("source_disjoint_from_calibration") is not True
        or pair_report.get("sealed_access") != SEALED_ZERO
        or Path(str(pair_report.get("output", ""))).resolve()
        != args.confirmation_pairs.resolve()
        or pair_report.get("output_sha256") != confirmation_pairs_sha256
        or application.get("pairs_sha256") != confirmation_pairs_sha256
        or application.get("pairs_report_sha256")
        != sha256_file(args.confirmation_pairs_report)
        or pair_report.get("inputs", {}).get("revision_report_sha256")
        != sha256_file(args.revision_report)
        or pair_report.get("inputs", {}).get("unchanged_report_sha256")
        != sha256_file(args.unchanged_report)
        or pair_report.get("inputs", {}).get("revision_candidates_sha256")
        != arm_receipts["revision"].get("candidates_sha256")
        or pair_report.get("inputs", {}).get("unchanged_candidates_sha256")
        != arm_receipts["unchanged"].get("candidates_sha256")
    ):
        raise PCF1ScoreError("PCF1 confirmation pair custody differs")

    prompt_truncated = application.get("prompt_truncated")
    malformed = application.get("malformed")
    maximum_swap_error = application.get("maximum_swap_error")
    if (
        application.get("rows") != TOTAL_ROWS
        or isinstance(malformed, bool)
        or not isinstance(malformed, int)
        or not 0 <= malformed <= TOTAL_ROWS
        or isinstance(prompt_truncated, bool)
        or not isinstance(prompt_truncated, int)
        or prompt_truncated < 0
        or isinstance(maximum_swap_error, bool)
        or not isinstance(maximum_swap_error, (int, float))
        or not math.isfinite(float(maximum_swap_error))
        or maximum_swap_error < 0
        or application.get("max_sequence_length") != 3072
        or application.get("sealed_access") != SEALED_ZERO
        or application.get("correctness_or_task_label_visible") is not False
        or application.get("protected_adapter_unchanged") is not True
        or application.get("commit_checkpoint_sha256")
        != training.get("checkpoint_sha256")
        or application.get("adapter_checkpoint_sha256")
        != training.get("adapter_checkpoint_sha256")
        or application.get("model_revision") != PINNED_MODEL_REVISION
        or application.get("model_revision") != training.get("model_revision")
        or application.get("model_root") != training.get("model_root")
        or application.get("model_loader") != "multimodal"
        or application.get("model_loader") != training.get("model_loader")
        or training.get("updates") != 128
        or training.get("gradient_accumulation") != 8
        or training.get("head_width") != 512
        or training.get("max_sequence_length") != 3072
        or training.get("seed") != 2026080822
        or training.get("backbone_learning_rate") != 2e-6
        or training.get("head_learning_rate") != 2e-4
        or training.get("protected_adapter_unchanged") is not True
        or training.get("sealed_access") != SEALED_ZERO
        or Path(str(application.get("selections", ""))).resolve()
        != args.selections.resolve()
        or application.get("selections_sha256") != sha256_file(args.selections)
    ):
        raise PCF1ScoreError("PCF1 commit application custody differs")
    selections = load_selections(args.selections)
    if application.get("order_consistent") != sum(
        int(row["order_consistent"]) for row in selections.values()
    ):
        raise PCF1ScoreError("PCF1 application/order receipt differs")

    identity_set = {str(row["identity_sha256"]) for row in sources}
    if any(set(rows) != identity_set for rows in (*candidates.values(), selections)):
        raise PCF1ScoreError("PCF1 confirmation identity sets differ")

    expected_hashes = _authorization_hashes(args)
    authorization = load_json(
        args.prescore_authorization,
        AUTHORIZATION_SCHEMA,
        "score authorization",
    )
    run_id = validate_authorization(
        authorization,
        expected_hashes,
        str(expected_assessors["sha256"]),
        identity_sha256,
        args.output_root,
        str(environment["environment_tree"]["sha256"]),
    )
    if (
        args.sandbox_probe_output.resolve()
        != score_sandbox_probe_path(args.output_root).resolve()
    ):
        raise PCF1ScoreError("PCF1 score authorization binding differs")

    authorization_sha256 = sha256_file(args.prescore_authorization)
    state.update(
        {
            "phase": "sandbox_qualification",
            "run_id": run_id,
            "authorization_sha256": authorization_sha256,
        }
    )
    sandbox_receipt = qualify_allocation()
    sandbox_probe_sha256 = str(sandbox_receipt["probe_sha256"])
    sandbox_receipt_sha256 = sandbox_atomic_json(
        args.sandbox_probe_output, sandbox_receipt
    )
    if (
        sandbox_receipt_sha256 != authorization.get("code_sandbox_probe_sha256")
        or sandbox_probe_sha256 != authorization.get("code_sandbox_probe_result_sha256")
        or sandbox_receipt_sha256 != authorization.get("code_sandbox_receipt_sha256")
    ):
        raise PCF1ScoreError("PCF1 final sandbox qualification differs")
    state["sandbox_probe_sha256"] = sandbox_receipt_sha256
    state["sandbox_probe_result_sha256"] = sandbox_probe_sha256
    state["sandbox_receipt_sha256"] = sandbox_receipt_sha256
    state["phase"] = "authorization_claim"
    consumption_path = score_consumption_path(args.output_root)
    consumption_payload = {
        "schema": CONSUMPTION_SCHEMA,
        "status": "complete",
        "claim_state": "consumed",
        "run_id": run_id,
        "score_output_root": str(args.output_root.resolve()),
        "score_authorization_sha256": authorization_sha256,
        "confirmation_assessors_sha256": str(expected_assessors["sha256"]),
        "identity_order_sha256": identity_sha256,
        "rows": TOTAL_ROWS,
        "semantic_read_budget": 1,
        "sandbox_probe_sha256": sandbox_receipt_sha256,
        "sandbox_probe_result_sha256": sandbox_probe_sha256,
        "sandbox_receipt_sha256": sandbox_receipt_sha256,
    }
    consumption_sha256 = consume_score_authorization(
        consumption_path, consumption_payload
    )
    state.update(
        {
            "claim_created": True,
            "consumption_sha256": consumption_sha256,
            "phase": "assessor_board_read",
        }
    )

    # No semantic assessor access is permitted above this line.
    assessors, assessor_sha256 = load_assessors_once(
        args.confirmation_assessors,
        str(expected_assessors["sha256"]),
        state,
    )
    if set(assessors) != identity_set:
        raise PCF1ScoreError("PCF1 assessor/source identity sets differ")

    state["phase"] = "assessor_setup_qualification"
    setup_qualifications = qualify_mbpp_assessor_setups(
        [assessors[str(source["identity_sha256"])]["assessor"] for source in sources]
    )
    state["sandbox_calls_started"] = int(state.get("sandbox_calls_started", 0)) + len(
        setup_qualifications
    )
    setup_qualifications_sha256 = mbpp_allocation_setup_receipts_sha256(
        setup_qualifications
    )

    correctness: dict[str, dict[str, bool]] = {arm: {} for arm in ARMS}
    malformed: dict[str, dict[str, bool]] = {arm: {} for arm in ARMS}
    capability_policy_rejected: dict[str, dict[str, bool]] = {arm: {} for arm in ARMS}
    score_completion_calls = 0
    assessment_rows: list[dict[str, Any]] = []
    state["phase"] = "candidate_assessment"
    for source in sources:
        identity = str(source["identity_sha256"])
        assessor_row = assessors[identity]
        if assessor_row.get("task") != source.get("task"):
            raise PCF1ScoreError("PCF1 assessor/source task binding differs")
        for arm in ARMS:
            completion = candidates[arm][identity]["completion"]
            is_malformed = not completion.strip()
            malformed[arm][identity] = is_malformed
            capability_policy_rejected[arm][identity] = False
            if is_malformed:
                correctness[arm][identity] = False
                continue
            state["assessment_calls_started"] = (
                int(state.get("assessment_calls_started", 0)) + 1
            )
            if source.get("task") == "mbpp":
                state["sandbox_calls_started"] = (
                    int(state.get("sandbox_calls_started", 0)) + 1
                )
            scored = score_completion(assessor_row["assessor"], completion)
            if not isinstance(scored, dict) or not isinstance(
                scored.get("correct"), bool
            ):
                raise PCF1ScoreError("PCF1 assessor result differs")
            execution = scored.get("execution")
            policy_rejected = isinstance(execution, dict) and (
                execution.get("candidate_policy_passed") is False
            )
            capability_policy_rejected[arm][identity] = policy_rejected
            malformed[arm][identity] = policy_rejected
            correctness[arm][identity] = scored["correct"]
            score_completion_calls += 1
        selection = selections[identity]
        selected_lineage = str(selection["selected_lineage"])
        assessment_rows.append(
            {
                "schema": OUTCOME_SCHEMA,
                "identity_sha256": identity,
                "task": source["task"],
                "revision_correct": correctness["revision"][identity],
                "unchanged_correct": correctness["unchanged"][identity],
                "self_refinement_correct": correctness["self_refinement"][identity],
                "revision_malformed": malformed["revision"][identity],
                "unchanged_malformed": malformed["unchanged"][identity],
                "self_refinement_malformed": malformed["self_refinement"][identity],
                "revision_capability_policy_rejected": capability_policy_rejected[
                    "revision"
                ][identity],
                "unchanged_capability_policy_rejected": capability_policy_rejected[
                    "unchanged"
                ][identity],
                "self_refinement_capability_policy_rejected": (
                    capability_policy_rejected["self_refinement"][identity]
                ),
                "selected_index": selection["selected_index"],
                "selected_lineage": selected_lineage,
                "selected_correct": correctness[selected_lineage][identity],
                "selected_malformed": malformed[selected_lineage][identity],
                "selected_capability_policy_rejected": capability_policy_rejected[
                    selected_lineage
                ][identity],
                "order_consistent": selection["order_consistent"],
                "score_consumption_sha256": consumption_sha256,
                "score_consumption_state": "consumed",
                "sandbox_probe_sha256": sandbox_receipt_sha256,
                "sandbox_probe_result_sha256": sandbox_probe_sha256,
                "sandbox_receipt_sha256": sandbox_receipt_sha256,
                "mbpp_allocation_setup_receipts_sha256": (setup_qualifications_sha256),
            }
        )

    metrics = {arm: arm_metrics(sources, correctness[arm]) for arm in ARMS}
    confirmation = commit_metrics(sources, correctness, selections)
    arm_malformed = {
        arm: sum(int(value) for value in malformed[arm].values()) for arm in ARMS
    }
    selected_malformed = sum(
        int(malformed[str(selection["selected_lineage"])][identity])
        for identity, selection in selections.items()
    )
    arm_capability_policy_rejected = {
        arm: sum(int(value) for value in capability_policy_rejected[arm].values())
        for arm in ARMS
    }
    selected_capability_policy_rejected = sum(
        int(capability_policy_rejected[str(selection["selected_lineage"])][identity])
        for identity, selection in selections.items()
    )
    result = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "run_id": run_id,
        "model_root": application["model_root"],
        "model_revision": application["model_revision"],
        "model_loader": application["model_loader"],
        "adapter_checkpoint_sha256": application["adapter_checkpoint_sha256"],
        "checkpoint_sha256": application["commit_checkpoint_sha256"],
        "training_report_sha256": sha256_file(args.training_report),
        "pairs_sha256": application["pairs_sha256"],
        "selections_sha256": application["selections_sha256"],
        "protected_adapter_unchanged": True,
        "max_sequence_length": application["max_sequence_length"],
        "arm_metrics": metrics,
        "confirmation": confirmation,
        "confirmation_prompt_truncated": prompt_truncated,
        "confirmation_malformed_selections": (
            application["malformed"] + selected_malformed
        ),
        "confirmation_malformed_candidates": sum(arm_malformed.values()),
        "arm_malformed": arm_malformed,
        "confirmation_capability_policy_rejections": sum(
            arm_capability_policy_rejected.values()
        ),
        "selected_capability_policy_rejections": (selected_capability_policy_rejected),
        "arm_capability_policy_rejected": arm_capability_policy_rejected,
        "confirmation_maximum_swap_error": maximum_swap_error,
        "assessment_calls": TOTAL_ROWS * len(ARMS),
        "score_completion_calls": score_completion_calls,
        "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "code_sandbox_binary_sha256": BWRAP_SHA256,
        "code_sandbox_probe_sha256": sandbox_receipt_sha256,
        "code_sandbox_probe_result_sha256": sandbox_probe_sha256,
        "code_sandbox_probe_passed": True,
        "sandbox_receipt": str(args.sandbox_probe_output.resolve()),
        "sandbox_receipt_sha256": sandbox_receipt_sha256,
        "mbpp_allocation_setup_status": "passed",
        "mbpp_allocation_setup_receipts": setup_qualifications,
        "mbpp_allocation_setup_receipt_count": len(setup_qualifications),
        "mbpp_allocation_setup_receipts_sha256": setup_qualifications_sha256,
        "environment_verified": True,
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": environment["environment_tree"]["sha256"],
        "assessor_board_sha256": assessor_sha256,
        "assessor_board_semantic_reads": 1,
        "confirmation_open_count": 1,
        "score_authorization_sha256": authorization_sha256,
        "score_consumption": str(consumption_path.resolve()),
        "score_consumption_sha256": consumption_sha256,
        "score_consumption_state": "consumed",
        "authorization_consumed": True,
        "inputs": {
            **expected_hashes,
            "confirmation_assessors_sha256": assessor_sha256,
        },
        "sealed_access": SEALED_ZERO,
    }
    state["phase"] = "atomic_score_publication"
    publish_score_root(args.output_root, assessment_rows, result)
    state["phase"] = "complete"
    return result


def score(args: argparse.Namespace) -> dict[str, Any]:
    """Run the one-shot scorer and preserve terminal post-claim failure evidence."""

    state: dict[str, Any] = {
        "phase": "prescore_validation",
        "assessor_bytes_read": 0,
        "assessor_rows_read": 0,
        "assessment_calls_started": 0,
        "sandbox_calls_started": 0,
        "claim_created": False,
    }
    try:
        return _score(args, state)
    except BaseException as error:
        if state.get("claim_created") is True:
            publish_terminal_failure(
                score_terminal_failure_path(args.output_root), args, state, error
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-data", type=Path, required=True)
    parser.add_argument("--confirmation-assessors", type=Path, required=True)
    parser.add_argument("--confirmation-assessor-receipt", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    for arm in ARMS:
        parser.add_argument(
            f"--{arm.replace('_', '-')}-report", type=Path, required=True
        )
        parser.add_argument(
            f"--{arm.replace('_', '-')}-candidates", type=Path, required=True
        )
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--confirmation-pairs", type=Path, required=True)
    parser.add_argument("--confirmation-pairs-report", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--application-report", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--data-custody", type=Path, required=True)
    parser.add_argument("--model-custody", type=Path, required=True)
    parser.add_argument("--runtime-custody", type=Path, required=True)
    parser.add_argument("--prescore-dispatch-receipt", type=Path, required=True)
    parser.add_argument("--prescore-accounting-receipt", type=Path, required=True)
    parser.add_argument("--prescore-authorization", type=Path, required=True)
    parser.add_argument("--sandbox-probe-output", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    result = score(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
