#!/usr/bin/env python3
"""Build Q36 precompute custody, one-shot authorization, or final custody."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

from build_pcf1_data import (
    CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA,
    DEVELOPMENT_SOURCE_SCHEMA,
    FREEZE_REPORT_SCHEMA,
    TRAIN_SOURCE_SCHEMA,
    _load_source_view,
    revision_prompt,
)
from build_q36_mtr_commit_pairs import REPORT_SCHEMA as PAIR_REPORT_SCHEMA
from build_q36_mtr_data import REPORT_SCHEMA as DATA_REPORT_SCHEMA
from compare_q36_mtr import ARM_SCHEMA, CUSTODY_SCHEMA
from hf_q36_mtr_evaluate import load_rows
from hf_q36_mtr_generate_drafts import SCHEMA as DRAFT_SCHEMA
from hf_q36_mtr_train_role import SCHEMA as ROLE_REPORT_SCHEMA
from merge_q36_mtr_drafts import SCHEMA as DRAFT_REPORT_SCHEMA
from merge_q36_mtr_evaluations import SCHEMA as EVALUATION_REPORT_SCHEMA
from pcf1_code_sandbox import BWRAP_SHA256, SANDBOX_CONFIG_SHA256
from q36_mtr_contract import MODEL_REVISION, STAGES, TOTAL_ROWS, validate_graph
from q36_mtr_roles import (
    MODEL_MANIFEST_SHA256,
    Q36MTRRoleError,
    REVISION_PRESENTATIONS,
    TRAINABLE_PARAMETERS,
    TRAINABLE_MASTER_DTYPE,
    role_contract,
    validate_matched_revision_geometry,
)
from score_q36_mtr import AUTHORIZATION_SCHEMA, CONSUMPTION_SCHEMA, SCORE_SCHEMA
from hf_q36_mtr_train_commit import (
    APPLICATION_SCHEMA,
    REPORT_SCHEMA as COMMIT_REPORT_SCHEMA,
)

PRECOMPUTE_SCHEMA = "shohin-q36-mtr-precompute-custody-v1"
ACCOUNTING_SCHEMA = "shohin-q36-mtr-slurm-accounting-v1"
EVIDENCE_SCHEMA = "shohin-q36-mtr-evidence-mirror-v1"
PRECOMPUTE_ARTIFACTS = {
    "aligned_checkpoint",
    "aligned_report",
    "application_report",
    "application_validation",
    "assessor_receipt",
    "calibration_data",
    "calibration_pairs",
    "calibration_pairs_report",
    "calibration_revision_candidates",
    "calibration_revision_report",
    "calibration_unchanged_candidates",
    "calibration_unchanged_report",
    "commit_checkpoint",
    "commit_training_report",
    "data_report",
    "development_data",
    "development_pairs",
    "development_pairs_report",
    "development_sources",
    "draft_hidden_candidates",
    "draft_hidden_evaluation_report",
    "draft_hidden_checkpoint",
    "draft_hidden_report",
    "draft_report",
    "drafts",
    "environment_receipt",
    "freeze_report",
    "live_preflight",
    "mechanics_report",
    "owner_checkpoint",
    "owner_data",
    "owner_report",
    "revision_candidates",
    "revision_report",
    "revision_training_data",
    "selections",
    "self_refinement_candidates",
    "self_refinement_report",
    "unchanged_candidates",
    "unchanged_report",
    "train_sources",
}
EVIDENCE_PRECOMPUTE_ARTIFACTS = PRECOMPUTE_ARTIFACTS - {
    "owner_data",
    "train_sources",
    "development_sources",
    "revision_training_data",
    "calibration_data",
    "development_data",
}
SCORE_HASH_MAPPING = {
    "development_data_sha256": "development_data",
    "data_report_sha256": "data_report",
    "assessor_receipt_sha256": "assessor_receipt",
    "application_report_sha256": "application_report",
    "selections_sha256": "selections",
    "commit_training_report_sha256": "commit_training_report",
    "environment_receipt_sha256": "environment_receipt",
    "prescore_accounting_sha256": "prescore_accounting",
}


class Q36MTRCustodyError(RuntimeError):
    """Q36 custody evidence is incomplete, inconsistent, or unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRCustodyError(f"refusing existing Q36 custody: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load(path: Path, schema: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRCustodyError(f"Q36 custody input is absent or symbolic: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or (
        schema is not None and value.get("schema") != schema
    ):
        raise Q36MTRCustodyError(f"Q36 custody schema differs: {path}")
    return value


def _artifacts(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, rendered = value.partition("=")
        path = Path(rendered)
        if not separator or not name or name in result or not path.is_absolute():
            raise Q36MTRCustodyError("Q36 explicit artifact binding differs")
        if path.is_symlink() or not path.is_file():
            raise Q36MTRCustodyError(f"Q36 artifact is absent or symbolic: {name}")
        result[name] = path.resolve(strict=True)
    return result


def _manifest_tree(root: Path, manifest: Path) -> dict[str, Any]:
    if (
        root.is_symlink()
        or not root.is_dir()
        or manifest.is_symlink()
        or not manifest.is_file()
    ):
        raise Q36MTRCustodyError("Q36 manifest root differs")
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        relative = relative[2:] if relative.startswith("./") else relative
        pure = PurePosixPath(relative)
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not pure.parts
            or pure.is_absolute()
            or "." in pure.parts
            or ".." in pure.parts
            or pure.as_posix() != relative
            or relative in {name for _, name in entries}
        ):
            raise Q36MTRCustodyError("Q36 manifest entry differs")
        path = root / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise Q36MTRCustodyError("Q36 manifest member differs")
        entries.append((digest, relative))
    actual = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            actual.add(path.relative_to(root).as_posix())
        elif not stat.S_ISDIR(mode):
            raise Q36MTRCustodyError("Q36 manifest tree has a link or special member")
    if actual != {name for _, name in entries} | {
        manifest.relative_to(root).as_posix()
    }:
        raise Q36MTRCustodyError("Q36 manifest exact membership differs")
    return {
        "manifest_sha256": sha256_file(manifest),
        "manifest_entries": len(entries),
        "regular_files": len(actual),
        "exact_membership": True,
    }


def _matches_file(report: dict[str, Any], field: str, path: Path) -> bool:
    return Path(str(report.get(field, ""))).resolve() == path.resolve() and report.get(
        f"{field}_sha256"
    ) == sha256_file(path)


def _validate_role_report(
    report: dict[str, Any], role: str, checkpoint: Path, owner_sha256: str
) -> None:
    expected = role_contract(role)
    selected = 100_000 if role == "owner" else 9_655
    expected_accumulation = 16 if role == "owner" else 8
    expected_consumed = 256 * expected_accumulation
    draft_bytes = role != "owner"
    draft_information = role == "aligned"
    consumption = report.get("training_consumption")
    if (
        report.get("schema") != ROLE_REPORT_SCHEMA
        or report.get("status") != "complete"
        or any(report.get(key) != value for key, value in expected.items())
        or report.get("update") != 256
        or report.get("updates") != 256
        or report.get("selected_rows") != selected
        or report.get("trainable_parameters") != TRAINABLE_PARAMETERS
        or report.get("trainable_master_dtype") != TRAINABLE_MASTER_DTYPE
        or report.get("trainable_compute_dtype") != "bfloat16"
        or not _matches_file(report, "checkpoint", checkpoint)
        or report.get("warm_start_checkpoint_sha256")
        != (None if role == "owner" else owner_sha256)
        or report.get("source_only_model_visible") is not (role == "owner")
        or report.get("internal_draft_visible") is not draft_information
        or report.get("draft_token_bytes_present") is not draft_bytes
        or report.get("draft_information_available") is not draft_information
        or report.get("draft_attention_applied") is not (role == "draft_hidden")
        or not isinstance(report.get("sequence_custody"), dict)
        or not isinstance(consumption, dict)
        or consumption.get("dataset_presentations") != selected
        or consumption.get("optimizer_updates") != 256
        or consumption.get("gradient_accumulation") != expected_accumulation
        or consumption.get("batch_size") != 1
        or consumption.get("microsteps") != expected_consumed
        or consumption.get("consumed_presentations") != expected_consumed
        or consumption.get("unique_consumed_presentations") != expected_consumed
        or consumption.get("complete_dataset_cycles") != 0
        or consumption.get("partial_cycle_presentations") != expected_consumed
        or any(
            not isinstance(consumption.get(field), str) or len(consumption[field]) != 64
            for field in (
                "presentation_index_sha256",
                "consumed_token_geometry_sha256",
                "consumed_draft_attention_sha256",
            )
        )
        or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
    ):
        raise Q36MTRCustodyError(f"Q36 {role} role custody differs")


def _validate_evaluation_report(
    report: dict[str, Any],
    arm: str,
    split: str,
    candidates: Path,
    checkpoint_sha256: str,
) -> None:
    rows = 5_824 if split == "calibration" else TOTAL_ROWS
    if (
        report.get("schema") != EVALUATION_REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("arm") != arm
        or report.get("split") != split
        or report.get("model_revision") != MODEL_REVISION
        or report.get("rows") != rows
        or (split == "development" and report.get("metrics") is not None)
        or (split == "calibration" and not isinstance(report.get("metrics"), dict))
        or report.get("adapter_checkpoint_sha256") != checkpoint_sha256
        or report.get("output_sha256") != sha256_file(candidates)
        or Path(str(report.get("output", ""))).resolve() != candidates.resolve()
        or report.get("exact_identity_coverage") is not True
        or report.get("duplicate_identities") != 0
        or report.get("assessor_board_access_count") != 0
        or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
    ):
        raise Q36MTRCustodyError(f"Q36 {arm} {split} evaluation differs")


def evaluation_checkpoint_sha256(arm: str, hashes: dict[str, str]) -> str:
    """Return the sole checkpoint authorized for a matched capability arm."""

    checkpoint_name = {
        "revision": "aligned_checkpoint",
        "unchanged": "owner_checkpoint",
        "self_refinement": "owner_checkpoint",
        "draft_hidden": "draft_hidden_checkpoint",
    }.get(arm)
    if checkpoint_name is None or checkpoint_name not in hashes:
        raise Q36MTRCustodyError(f"Q36 {arm} checkpoint lineage differs")
    return hashes[checkpoint_name]


def validate_causal_intervention_receipt(mechanics: dict[str, Any]) -> None:
    """Require measured state and native-router causality before custody."""

    causal = mechanics.get("causal_draft_intervention")
    native = causal.get("native_router") if isinstance(causal, dict) else None
    aligned = native.get("aligned") if isinstance(native, dict) else None
    hidden = native.get("draft_hidden") if isinstance(native, dict) else None

    def finite(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    valid = (
        isinstance(causal, dict)
        and causal.get("token_count_exact") is True
        and causal.get("position_geometry_exact") is True
        and causal.get("draft_hidden_counterfactual_invariant") is True
        and causal.get("aligned_counterfactual_sensitive") is True
        and causal.get("draft_hidden_invariant_tolerance") == 2e-3
        and causal.get("aligned_sensitivity_floor") == 1e-2
        and finite(causal.get("aligned_response_max_abs_delta"))
        and float(causal["aligned_response_max_abs_delta"]) >= 1e-2
        and finite(causal.get("draft_hidden_response_max_abs_delta"))
        and float(causal["draft_hidden_response_max_abs_delta"]) <= 2e-3
        and isinstance(native, dict)
        and native.get("invariant_tolerance") == 2e-3
        and native.get("sensitivity_floor") == 1e-2
        and native.get("draft_hidden_route_invariant") is True
        and native.get("aligned_route_sensitive") is True
        and isinstance(native.get("aligned_expert_selection_changed"), bool)
        and isinstance(aligned, dict)
        and isinstance(hidden, dict)
        and isinstance(aligned.get("layers"), int)
        and not isinstance(aligned.get("layers"), bool)
        and aligned["layers"] > 0
        and hidden.get("layers") == aligned["layers"]
        and isinstance(aligned.get("top_k"), int)
        and not isinstance(aligned.get("top_k"), bool)
        and aligned["top_k"] > 0
        and hidden.get("top_k") == aligned["top_k"]
        and isinstance(aligned.get("topk_assignment_changes"), int)
        and aligned["topk_assignment_changes"] >= 0
        and hidden.get("topk_assignment_changes") == 0
        and native["aligned_expert_selection_changed"]
        is (aligned["topk_assignment_changes"] > 0)
        and finite(aligned.get("router_max_abs_delta"))
        and float(aligned["router_max_abs_delta"]) >= 1e-2
        and finite(hidden.get("router_max_abs_delta"))
        and float(hidden["router_max_abs_delta"]) <= 2e-3
        and all(
            isinstance(route.get("route_path_sha256"), str)
            and len(route["route_path_sha256"]) == 64
            for route in (aligned, hidden)
        )
    )
    if not valid:
        raise Q36MTRCustodyError("Q36 causal intervention custody differs")


def validate_draft_byte_custody(
    artifacts: dict[str, Path],
    data_report: dict[str, Any],
    train_sources: dict[str, dict[str, Any]],
    development_sources: dict[str, dict[str, Any]],
) -> None:
    """Replay raw decode to canonical prompt bytes for every identity."""

    drafts: dict[str, str] = {}
    receipts: list[dict[str, str]] = []
    canonicalized = 0
    sources = {**train_sources, **development_sources}
    for line in artifacts["drafts"].read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        identity = str(row.get("identity_sha256", ""))
        raw = row.get("completion")
        source = sources.get(identity)
        if (
            identity in drafts
            or source is None
            or row.get("schema") != DRAFT_SCHEMA
            or row.get("split") != source["split"]
            or row.get("task") != source["task"]
            or row.get("prompt_sha256")
            != hashlib.sha256(str(source["source_prompt"]).encode()).hexdigest()
            or row.get("owner_checkpoint_sha256")
            != data_report.get("owner_checkpoint_sha256")
            or row.get("model_revision") != MODEL_REVISION
            or not isinstance(raw, str)
            or not raw.strip()
        ):
            raise Q36MTRCustodyError("Q36 raw draft byte custody differs")
        canonical = raw.strip()
        raw_sha256 = hashlib.sha256(raw.encode()).hexdigest()
        canonical_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
        drafts[identity] = raw
        canonicalized += int(raw != canonical)
        receipts.append(
            {
                "identity_sha256": identity,
                "raw_sha256": raw_sha256,
                "canonical_sha256": canonical_sha256,
            }
        )
    if set(drafts) != set(sources):
        raise Q36MTRCustodyError("Q36 raw draft identity custody differs")

    def validate_row(row: dict[str, Any], identity: str) -> None:
        raw = drafts[identity]
        canonical = raw.strip()
        draft = row.get("internal_draft")
        if (
            row.get("model_owned_draft_sha256")
            != hashlib.sha256(canonical.encode()).hexdigest()
            or row.get("raw_model_owned_draft_sha256")
            != hashlib.sha256(raw.encode()).hexdigest()
            or row.get("draft_canonicalization") != "unicode_outer_whitespace_strip_v1"
            or not isinstance(draft, dict)
            or draft.get("completion") != canonical
            or row.get("question")
            != revision_prompt(str(sources[identity]["source_prompt"]), canonical)
        ):
            raise Q36MTRCustodyError("Q36 canonical draft prompt custody differs")

    for split, name in (
        ("calibration", "calibration_data"),
        ("development", "development_data"),
    ):
        for row in load_rows(artifacts[name], split):
            validate_row(row, str(row["identity_sha256"]))
    revision_rows = 0
    for line in (
        artifacts["revision_training_data"].read_text(encoding="utf-8").splitlines()
    ):
        if not line:
            continue
        row = json.loads(line)
        identity = str(row.get("source_identity_sha256", ""))
        if identity not in train_sources:
            raise Q36MTRCustodyError("Q36 revision draft identity differs")
        raw = drafts[identity]
        canonical = raw.strip()
        if (
            row.get("model_owned_draft_sha256")
            != hashlib.sha256(canonical.encode()).hexdigest()
            or row.get("raw_model_owned_draft_sha256")
            != hashlib.sha256(raw.encode()).hexdigest()
            or row.get("draft_canonicalization") != "unicode_outer_whitespace_strip_v1"
            or row.get("question")
            != revision_prompt(str(train_sources[identity]["source_prompt"]), canonical)
        ):
            raise Q36MTRCustodyError("Q36 revision canonical draft differs")
        revision_rows += 1
    byte_custody = data_report.get("draft_byte_custody")
    digest = hashlib.sha256(
        b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for row in sorted(receipts, key=lambda value: value["identity_sha256"])
        )
    ).hexdigest()
    if (
        revision_rows != REVISION_PRESENTATIONS
        or not isinstance(byte_custody, dict)
        or byte_custody.get("raw_decode_preserved_in_merged_drafts") is not True
        or byte_custody.get("canonicalization") != "unicode_outer_whitespace_strip_v1"
        or byte_custody.get("canonicalized_drafts") != canonicalized
        or byte_custody.get("identity_raw_canonical_sha256") != digest
    ):
        raise Q36MTRCustodyError("Q36 draft byte receipt differs")


def _validate_precompute_lineage(artifacts: dict[str, Path]) -> None:
    hashes = {name: sha256_file(path) for name, path in artifacts.items()}
    owner = _load(artifacts["owner_report"], ROLE_REPORT_SCHEMA)
    aligned = _load(artifacts["aligned_report"], ROLE_REPORT_SCHEMA)
    hidden = _load(artifacts["draft_hidden_report"], ROLE_REPORT_SCHEMA)
    owner_sha256 = hashes["owner_checkpoint"]
    _validate_role_report(owner, "owner", artifacts["owner_checkpoint"], owner_sha256)
    _validate_role_report(
        aligned, "aligned", artifacts["aligned_checkpoint"], owner_sha256
    )
    _validate_role_report(
        hidden, "draft_hidden", artifacts["draft_hidden_checkpoint"], owner_sha256
    )
    try:
        validate_matched_revision_geometry(
            aligned["sequence_custody"], hidden["sequence_custody"]
        )
    except (KeyError, Q36MTRRoleError) as error:
        raise Q36MTRCustodyError(
            "Q36 aligned/hidden causal geometry differs"
        ) from error
    if aligned["training_consumption"] != hidden["training_consumption"]:
        raise Q36MTRCustodyError("Q36 aligned/hidden consumed prefix differs")
    freeze = _load(artifacts["freeze_report"], FREEZE_REPORT_SCHEMA)
    draft = _load(artifacts["draft_report"], DRAFT_REPORT_SCHEMA)
    data = _load(artifacts["data_report"], DATA_REPORT_SCHEMA)
    train_sources = _load_source_view(
        artifacts["train_sources"], TRAIN_SOURCE_SCHEMA, "train"
    )
    development_sources = _load_source_view(
        artifacts["development_sources"], DEVELOPMENT_SOURCE_SCHEMA, "development"
    )
    validate_draft_byte_custody(artifacts, data, train_sources, development_sources)
    if (
        freeze.get("status") != "complete"
        or freeze.get("source_disjoint") is not True
        or freeze.get("sealed_content_materialized") is not False
        or len(train_sources) != 5_824
        or len(development_sources) != TOTAL_ROWS
        or draft.get("status") != "complete"
        or draft.get("rows") != 7_113
        or draft.get("owner_checkpoint_sha256") != owner_sha256
        or not _matches_file(draft, "output", artifacts["drafts"])
        or data.get("draft_report_sha256") != hashes["draft_report"]
        or data.get("drafts_sha256") != hashes["drafts"]
        or data.get("freeze_report_sha256") != hashes["freeze_report"]
        or data.get("source_disjoint") is not True
        or data.get("model_owned_drafts") is not True
        or data.get("sealed_content_materialized") is not False
        or data.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or owner.get("data_sha256") != hashes["owner_data"]
        or aligned.get("data_sha256") != hashes["revision_training_data"]
        or hidden.get("data_sha256") != hashes["revision_training_data"]
    ):
        raise Q36MTRCustodyError("Q36 source/draft/data lineage differs")
    expected_outputs = {
        "revision_train": "revision_training_data",
        "calibration": "calibration_data",
        "development": "development_data",
    }
    for key, name in expected_outputs.items():
        output = data.get("outputs", {}).get(key)
        if (
            not isinstance(output, dict)
            or output.get("sha256") != hashes[name]
            or Path(str(output.get("path", ""))).resolve() != artifacts[name].resolve()
        ):
            raise Q36MTRCustodyError(f"Q36 materialized {key} lineage differs")
    for arm in ("revision", "unchanged", "self_refinement", "draft_hidden"):
        report_name = (
            "draft_hidden_evaluation_report"
            if arm == "draft_hidden"
            else f"{arm}_report"
        )
        _validate_evaluation_report(
            _load(artifacts[report_name], EVALUATION_REPORT_SCHEMA),
            arm,
            "development",
            artifacts[f"{arm}_candidates"],
            evaluation_checkpoint_sha256(arm, hashes),
        )
    for arm in ("revision", "unchanged"):
        _validate_evaluation_report(
            _load(artifacts[f"calibration_{arm}_report"], EVALUATION_REPORT_SCHEMA),
            arm,
            "calibration",
            artifacts[f"calibration_{arm}_candidates"],
            evaluation_checkpoint_sha256(arm, hashes),
        )
    for split, pair_name, report_name, rows in (
        ("calibration", "calibration_pairs", "calibration_pairs_report", 5_824),
        ("development", "development_pairs", "development_pairs_report", TOTAL_ROWS),
    ):
        report = _load(artifacts[report_name], PAIR_REPORT_SCHEMA)
        if (
            report.get("status") != "complete"
            or report.get("model_revision") != MODEL_REVISION
            or report.get("source_split") != split
            or report.get("rows") != rows
            or report.get("output_sha256") != hashes[pair_name]
            or Path(str(report.get("output", ""))).resolve()
            != artifacts[pair_name].resolve()
            or report.get("assessor_board_access_count") != 0
            or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
            or (
                split == "development"
                and report.get("labels_or_correctness_fields") != 0
            )
        ):
            raise Q36MTRCustodyError(f"Q36 {split} commit-pair custody differs")
        source_name = (
            "calibration_data" if split == "calibration" else "development_data"
        )
        prefix = "calibration_" if split == "calibration" else ""
        inputs = report.get("inputs", {})
        if (
            inputs.get("data_sha256") != hashes[source_name]
            or inputs.get("revision_report_sha256")
            != hashes[f"{prefix}revision_report"]
            or inputs.get("revision_candidates_sha256")
            != hashes[f"{prefix}revision_candidates"]
            or inputs.get("unchanged_report_sha256")
            != hashes[f"{prefix}unchanged_report"]
            or inputs.get("unchanged_candidates_sha256")
            != hashes[f"{prefix}unchanged_candidates"]
        ):
            raise Q36MTRCustodyError(f"Q36 {split} commit-pair inputs differ")
    application = _load(artifacts["application_report"], APPLICATION_SCHEMA)
    application_validation = _load(
        artifacts["application_validation"],
        "shohin-q36-mtr-commit-application-validation-v1",
    )
    commit = _load(artifacts["commit_training_report"], COMMIT_REPORT_SCHEMA)
    if (
        commit.get("status") != "complete"
        or commit.get("model_revision") != MODEL_REVISION
        or not _matches_file(commit, "checkpoint", artifacts["commit_checkpoint"])
        or commit.get("protected_adapter_sha256_after") != hashes["aligned_checkpoint"]
        or commit.get("protected_adapter_unchanged") is not True
        or commit.get("trainable_master_dtype") != TRAINABLE_MASTER_DTYPE
        or commit.get("trainable_compute_dtype") != "bfloat16"
        or Path(str(commit.get("development_application_report", ""))).resolve()
        != artifacts["application_report"].resolve()
        or commit.get("development_selections_sha256") != hashes["selections"]
        or application.get("status") != "complete"
        or application.get("commit_checkpoint_sha256") != hashes["commit_checkpoint"]
        or application.get("selections_sha256") != hashes["selections"]
        or Path(str(application.get("selections", ""))).resolve()
        != artifacts["selections"].resolve()
        or application.get("assessor_board_access_count") != 0
        or application.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or application_validation.get("status") != "complete"
        or application_validation.get("commit_checkpoint_sha256")
        != hashes["commit_checkpoint"]
        or application_validation.get("commit_training_report_sha256")
        != hashes["commit_training_report"]
        or application_validation.get("development_pairs_sha256")
        != hashes["development_pairs"]
        or application_validation.get("development_pairs_report_sha256")
        != hashes["development_pairs_report"]
        or application_validation.get("application_report_sha256")
        != hashes["application_report"]
        or application_validation.get("selections_sha256") != hashes["selections"]
        or application_validation.get("assessor_board_access_count") != 0
        or application_validation.get("sealed_access")
        != {"holdout": 0, "product": 0, "public": 0}
    ):
        raise Q36MTRCustodyError("Q36 learned-commit lineage differs")


def _hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_checkpoint_hashes(artifacts: dict[str, Path]) -> dict[str, str]:
    hashes = {name: sha256_file(path) for name, path in artifacts.items()}
    expected = {
        "owner_report": ("owner_checkpoint", "checkpoint_sha256"),
        "aligned_report": ("aligned_checkpoint", "checkpoint_sha256"),
        "draft_hidden_report": ("draft_hidden_checkpoint", "checkpoint_sha256"),
        "commit_training_report": ("commit_checkpoint", "checkpoint_sha256"),
    }
    for report_name, (checkpoint_name, field) in expected.items():
        report = _load(artifacts[report_name])
        if report.get(field) != hashes[checkpoint_name]:
            raise Q36MTRCustodyError(
                f"Q36 {checkpoint_name} hash differs from its report"
            )
    return hashes


def build_precompute(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = _artifacts(args.artifact)
    if set(artifacts) != PRECOMPUTE_ARTIFACTS:
        raise Q36MTRCustodyError("Q36 precompute artifact set differs")
    _validate_precompute_lineage(artifacts)
    graph = _load(args.graph_contract)
    validate_graph(graph)
    runtime = _manifest_tree(args.runtime_root, args.runtime_manifest)
    model = _manifest_tree(args.model_root, args.model_manifest)
    runtime_receipt = _load(args.runtime_root / "runtime.json")
    if (
        runtime_receipt.get("schema") != "shohin-q36-mtr-runtime-v1"
        or runtime_receipt.get("status") != "complete"
        or runtime_receipt.get("source_commit") != graph.get("source_commit")
    ):
        raise Q36MTRCustodyError("Q36 runtime/source commit binding differs")
    development = load_rows(artifacts["development_data"], "development")
    identity_order_sha256 = hashlib.sha256(
        ("\n".join(str(row["identity_sha256"]) for row in development) + "\n").encode()
    ).hexdigest()
    data_report = _load(artifacts["data_report"], "shohin-q36-mtr-data-report-v1")
    assessor = _load(
        artifacts["assessor_receipt"], CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA
    )
    mechanics = _load(artifacts["mechanics_report"], "shohin-q36-mtr-mechanics-v1")
    environment = _load(
        artifacts["environment_receipt"], "shohin-q36-mtr-environment-v1"
    )
    live_preflight = _load(
        artifacts["live_preflight"], "shohin-q36-mtr-live-preflight-v1"
    )
    validate_causal_intervention_receipt(mechanics)
    if (
        data_report.get("model_revision") != MODEL_REVISION
        or data_report.get("outputs", {}).get("development", {}).get("sha256")
        != sha256_file(artifacts["development_data"])
        or data_report.get("source_disjoint") is not True
        or assessor.get("rows") != TOTAL_ROWS
        or assessor.get("semantic_access") != "final_score_only"
        or mechanics.get("status") != "pass"
        or mechanics.get("capability_scored") is not False
        or mechanics.get("trainable_parameters") != 1_179_648
        or mechanics.get("protected_router_expert_trainables") != 0
        or mechanics.get("protected_parameter_receipt_before")
        != mechanics.get("protected_parameter_receipt_after")
        or mechanics.get("one_finite_update") is not True
        or mechanics.get("serialization_restore_exact") is not True
        or environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("runtime_manifest_sha256") != runtime["manifest_sha256"]
        or live_preflight.get("status") != "pass"
        or live_preflight.get("run_id") != args.run_id
        or live_preflight.get("source_commit") != graph.get("source_commit")
        or live_preflight.get("graph_contract_sha256")
        != sha256_file(args.graph_contract)
        or live_preflight.get("environment_receipt_sha256")
        != sha256_file(artifacts["environment_receipt"])
        or live_preflight.get("model_revision") != MODEL_REVISION
        or live_preflight.get("scientific_rows_read") != 0
        or live_preflight.get("capability_scored") is not False
        or live_preflight.get("scientific_jobs_submitted_by_preflight") != 0
        or live_preflight.get("automatic_retry") is not False
        or live_preflight.get("automatic_successor") is not False
        or live_preflight.get("sealed_access")
        != {"holdout": 0, "product": 0, "public": 0}
        or model.get("manifest_sha256") != MODEL_MANIFEST_SHA256
    ):
        raise Q36MTRCustodyError("Q36 precompute scientific custody differs")
    artifact_hashes = dict(sorted(_validate_checkpoint_hashes(artifacts).items()))
    payload = {
        "schema": PRECOMPUTE_SCHEMA,
        "status": "complete",
        "run_id": args.run_id,
        "source_commit": graph["source_commit"],
        "graph_contract_sha256": sha256_file(args.graph_contract),
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": model["manifest_sha256"],
        "runtime_sha256": runtime["manifest_sha256"],
        "runtime_manifest_sha256": runtime["manifest_sha256"],
        "environment_receipt_sha256": artifact_hashes["environment_receipt"],
        "live_preflight_sha256": artifact_hashes["live_preflight"],
        "sandbox_receipt_sha256": mechanics.get("sandbox_receipt_sha256"),
        "data_sha256": artifact_hashes["development_data"],
        "identity_order_sha256": identity_order_sha256,
        "assessor_board_sha256": assessor["board_sha256"],
        "assessor_receipt_sha256": artifact_hashes["assessor_receipt"],
        "assessor_semantic_reads": 0,
        "artifact_sha256s": artifact_hashes,
        "model_manifest_verified": True,
        "runtime_manifest_verified": True,
        "environment_verified": True,
        "mechanics_verified": True,
        "source_disjoint": True,
        "custody_verified": True,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.output, payload)
    return payload


def build_authorization(args: argparse.Namespace) -> dict[str, Any]:
    precompute = _load(args.precompute_custody, PRECOMPUTE_SCHEMA)
    artifacts = _artifacts(args.artifact)
    required = {
        "application_report",
        "assessor_receipt",
        "commit_training_report",
        "data_report",
        "development_data",
        "draft_hidden_candidates",
        "draft_hidden_evaluation_report",
        "environment_receipt",
        "prescore_accounting",
        "revision_candidates",
        "revision_report",
        "selections",
        "self_refinement_candidates",
        "self_refinement_report",
        "unchanged_candidates",
        "unchanged_report",
    }
    if set(artifacts) != required:
        raise Q36MTRCustodyError("Q36 authorization artifact set differs")
    for name, path in artifacts.items():
        if name == "prescore_accounting":
            continue
        if precompute.get("artifact_sha256s", {}).get(name) != sha256_file(path):
            raise Q36MTRCustodyError("Q36 authorization/precompute hash differs")
    graph_sha256 = sha256_file(args.graph_contract)
    if precompute.get("graph_contract_sha256") != graph_sha256:
        raise Q36MTRCustodyError("Q36 authorization graph differs")
    prescore = _load(artifacts["prescore_accounting"], ACCOUNTING_SCHEMA)
    prescore_required = [stage.name for stage in STAGES]
    prescore_required = prescore_required[
        : prescore_required.index("precompute_custody") + 1
    ]
    if (
        prescore.get("status") != "complete"
        or prescore.get("phase") != "prescore"
        or prescore.get("run_id") != precompute.get("run_id")
        or prescore.get("source_commit") != precompute.get("source_commit")
        or prescore.get("graph_contract_sha256") != graph_sha256
        or prescore.get("required_stages") != prescore_required
        or prescore.get("h100_request_count") != 61
        or prescore.get("completed_h100_allocation_count") != 61
        or prescore.get("retry_count") != 0
        or prescore.get("requeue_count") != 0
        or prescore.get("duplicate_shard_count") != 0
        or prescore.get("orphaned_job_count") != 0
        or prescore.get("successor_authorized") is not False
        or prescore.get("successor_submitted") is not False
        or prescore.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
    ):
        raise Q36MTRCustodyError("Q36 prescore accounting differs")
    input_hashes = {
        name: sha256_file(artifacts[source])
        for name, source in SCORE_HASH_MAPPING.items()
    }
    input_hashes["arm_report_sha256s"] = {
        "revision": sha256_file(artifacts["revision_report"]),
        "unchanged": sha256_file(artifacts["unchanged_report"]),
        "self_refinement": sha256_file(artifacts["self_refinement_report"]),
        "draft_hidden": sha256_file(artifacts["draft_hidden_evaluation_report"]),
    }
    input_hashes["arm_candidate_sha256s"] = {
        arm: sha256_file(artifacts[f"{arm}_candidates"])
        for arm in ("revision", "unchanged", "self_refinement", "draft_hidden")
    }
    input_hashes["precompute_custody_sha256"] = sha256_file(args.precompute_custody)
    input_hashes["graph_contract_sha256"] = graph_sha256
    payload = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "complete",
        "run_id": precompute["run_id"],
        "scoring_authorized": True,
        "one_shot": True,
        "model_revision": MODEL_REVISION,
        "rows": TOTAL_ROWS,
        "identity_order_sha256": precompute["identity_order_sha256"],
        "assessor_board_sha256": precompute["assessor_board_sha256"],
        "score_output_root": str(args.score_output_root.resolve()),
        "assessor_board_access_count_before": 0,
        "input_hashes": input_hashes,
        "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "code_sandbox_binary_sha256": BWRAP_SHA256,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        "automatic_retry": False,
        "automatic_successor": False,
    }
    _atomic_json(args.output, payload)
    return payload


def build_final(args: argparse.Namespace) -> dict[str, Any]:
    precompute = _load(args.precompute_custody, PRECOMPUTE_SCHEMA)
    score = _load(args.score_report, SCORE_SCHEMA)
    consumption = _load(args.score_consumption, CONSUMPTION_SCHEMA)
    accounting = _load(args.scheduler_accounting, ACCOUNTING_SCHEMA)
    evidence = _load(args.evidence_mirror, EVIDENCE_SCHEMA)
    graph = _load(args.graph_contract)
    validate_graph(graph)
    arm_paths = _artifacts(args.arm_report)
    expected_arms = {
        "learned_commit",
        "trained_revision",
        "unchanged",
        "self_refinement",
        "draft_hidden",
    }
    if set(arm_paths) != expected_arms:
        raise Q36MTRCustodyError("Q36 final arm set differs")
    precompute_sha256 = sha256_file(args.precompute_custody)
    score_sha256 = sha256_file(args.score_report)
    consumption_sha256 = sha256_file(args.score_consumption)
    accounting_sha256 = sha256_file(args.scheduler_accounting)
    graph_sha256 = sha256_file(args.graph_contract)
    arm_sha256s = {name: sha256_file(path) for name, path in sorted(arm_paths.items())}
    for name, path in arm_paths.items():
        report = _load(path, ARM_SCHEMA)
        if (
            report.get("status") != "complete"
            or report.get("arm") != name
            or report.get("split") != "development"
            or report.get("run_id") != precompute["run_id"]
            or report.get("model_revision") != MODEL_REVISION
            or report.get("full_row_count") != TOTAL_ROWS
            or report.get("candidate_count") != TOTAL_ROWS
            or report.get("identity_order_sha256")
            != precompute["identity_order_sha256"]
            or report.get("data_sha256") != precompute["data_sha256"]
            or report.get("runtime_sha256") != precompute["runtime_sha256"]
            or report.get("precompute_custody_sha256") != precompute_sha256
            or report.get("score_report_sha256") != score_sha256
            or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        ):
            raise Q36MTRCustodyError("Q36 final arm binding differs")
    evidence_hashes = evidence.get("artifact_sha256s")
    expected_evidence_hashes = {
        "graph_contract": graph_sha256,
        "precompute_custody": precompute_sha256,
        "score_report": score_sha256,
        "score_consumption": consumption_sha256,
        "scheduler_accounting": accounting_sha256,
        "prescore_accounting": score["input_hashes"]["prescore_accounting_sha256"],
        "score_authorization": score["score_authorization_sha256"],
        "score_outcomes": score["outcomes_sha256"],
        "score_sandbox_receipt": score["sandbox_receipt_sha256"],
        "plan": accounting["plan_sha256"],
        "dispatch_receipt": accounting["dispatch_receipt_sha256"],
        "model_manifest": precompute["model_manifest_sha256"],
        "runtime_manifest": precompute["runtime_manifest_sha256"],
        **{f"arm_{name}": digest for name, digest in arm_sha256s.items()},
        **{
            f"precompute_{name}": precompute["artifact_sha256s"][name]
            for name in sorted(EVIDENCE_PRECOMPUTE_ARTIFACTS)
        },
    }
    if (
        precompute.get("status") != "complete"
        or precompute.get("source_commit") != graph.get("source_commit")
        or precompute.get("graph_contract_sha256") != graph_sha256
        or score.get("status") != "complete"
        or score.get("run_id") != precompute["run_id"]
        or score.get("model_revision") != MODEL_REVISION
        or score.get("rows") != TOTAL_ROWS
        or score.get("outcome_rows") != TOTAL_ROWS
        or score.get("identity_order_sha256") != precompute["identity_order_sha256"]
        or score.get("score_consumption_sha256") != consumption_sha256
        or score.get("score_consumption_state") != "consumed"
        or score.get("assessor_semantic_reads") != 1
        or score.get("assessor_rows_read") != TOTAL_ROWS
        or score.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or not _hex(score.get("sandbox_receipt_sha256"))
        or not _hex(score.get("sandbox_probe_sha256"))
        or consumption.get("status") != "consumed"
        or consumption.get("run_id") != precompute["run_id"]
        or consumption.get("authorization_sha256")
        != score.get("score_authorization_sha256")
        or consumption.get("score_output_root")
        != str(args.score_report.resolve().parent)
        or accounting.get("status") != "complete"
        or accounting.get("phase") != "final"
        or accounting.get("run_id") != precompute["run_id"]
        or accounting.get("source_commit") != graph.get("source_commit")
        or accounting.get("graph_contract_sha256") != graph_sha256
        or accounting.get("required_stages")
        != [stage.name for stage in STAGES][
            : [stage.name for stage in STAGES].index("normalize") + 1
        ]
        or accounting.get("h100_request_count") != 61
        or accounting.get("completed_h100_allocation_count") != 61
        or accounting.get("retry_count") != 0
        or accounting.get("requeue_count") != 0
        or accounting.get("duplicate_shard_count") != 0
        or accounting.get("orphaned_job_count") != 0
        or accounting.get("successor_authorized") is not False
        or accounting.get("successor_submitted") is not False
        or not isinstance(accounting.get("charged_gpu_seconds"), (int, float))
        or accounting["charged_gpu_seconds"] <= 0
        or evidence.get("status") != "complete"
        or evidence.get("verified") is not True
        or evidence.get("run_id") != precompute["run_id"]
        or evidence.get("source_commit") != graph.get("source_commit")
        or evidence.get("graph_contract_sha256") != graph_sha256
        or evidence_hashes != expected_evidence_hashes
    ):
        raise Q36MTRCustodyError("Q36 final execution custody differs")
    artifact_hashes = precompute["artifact_sha256s"]
    payload = {
        "schema": CUSTODY_SCHEMA,
        "status": "complete",
        "run_id": precompute["run_id"],
        "source_commit": graph["source_commit"],
        "graph_contract_sha256": graph_sha256,
        "model_revision": MODEL_REVISION,
        "model_manifest_verified": True,
        "model_manifest_sha256": precompute["model_manifest_sha256"],
        "runtime_manifest_verified": True,
        "runtime_manifest_sha256": precompute["runtime_manifest_sha256"],
        "runtime_source_commit": graph["source_commit"],
        "runtime_sha256": precompute["runtime_sha256"],
        "data_sha256": precompute["data_sha256"],
        "identity_order_sha256": precompute["identity_order_sha256"],
        "precompute_custody_sha256": precompute_sha256,
        "arm_report_sha256s": arm_sha256s,
        "checkpoint_hashes_verified": True,
        "checkpoint_sha256s": {
            "owner": artifact_hashes["owner_checkpoint"],
            "trained_revision": artifact_hashes["aligned_checkpoint"],
            "draft_hidden": artifact_hashes["draft_hidden_checkpoint"],
            "learned_commit": artifact_hashes["commit_checkpoint"],
        },
        "environment_verified": True,
        "environment_receipt_sha256": precompute["environment_receipt_sha256"],
        "sandbox_verified": True,
        "sandbox_receipt_sha256": score["sandbox_receipt_sha256"],
        "scheduler_accounting_verified": True,
        "scheduler_accounting_sha256": accounting_sha256,
        "one_assessor_open_verified": True,
        "assessor_semantic_reads": score["assessor_semantic_reads"],
        "public_access_count": 0,
        "holdout_access_count": 0,
        "product_access_count": 0,
        "retry_count": accounting["retry_count"],
        "requeue_count": accounting["requeue_count"],
        "duplicate_shard_count": accounting["duplicate_shard_count"],
        "orphaned_job_count": accounting["orphaned_job_count"],
        "successor_authorized": False,
        "successor_submitted": False,
        "score_consumption_state": "consumed",
        "score_consumption_sha256": consumption_sha256,
        "h100_request_count": accounting["h100_request_count"],
        "completed_h100_allocation_count": accounting[
            "completed_h100_allocation_count"
        ],
        "charged_gpu_seconds": accounting["charged_gpu_seconds"],
        "evidence_mirror_verified": True,
        "evidence_mirror_manifest_sha256": sha256_file(args.evidence_mirror),
        "source_disjoint": True,
        "custody_verified": True,
    }
    _atomic_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    precompute = modes.add_parser("precompute")
    precompute.add_argument("--run-id", required=True)
    precompute.add_argument("--graph-contract", type=Path, required=True)
    precompute.add_argument("--runtime-root", type=Path, required=True)
    precompute.add_argument("--runtime-manifest", type=Path, required=True)
    precompute.add_argument("--model-root", type=Path, required=True)
    precompute.add_argument("--model-manifest", type=Path, required=True)
    precompute.add_argument("--artifact", action="append", default=[])
    precompute.add_argument("--output", type=Path, required=True)
    authorize = modes.add_parser("authorize")
    authorize.add_argument("--precompute-custody", type=Path, required=True)
    authorize.add_argument("--graph-contract", type=Path, required=True)
    authorize.add_argument("--artifact", action="append", default=[])
    authorize.add_argument("--score-output-root", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)
    final = modes.add_parser("final")
    final.add_argument("--precompute-custody", type=Path, required=True)
    final.add_argument("--score-report", type=Path, required=True)
    final.add_argument("--score-consumption", type=Path, required=True)
    final.add_argument("--scheduler-accounting", type=Path, required=True)
    final.add_argument("--evidence-mirror", type=Path, required=True)
    final.add_argument("--graph-contract", type=Path, required=True)
    final.add_argument("--arm-report", action="append", default=[])
    final.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = {
        "precompute": build_precompute,
        "authorize": build_authorization,
        "final": build_final,
    }[args.mode](args)
    print(json.dumps({"schema": result["schema"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
