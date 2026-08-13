#!/usr/bin/env python3
"""Emit and validate the frozen, no-submit Q36-MTR execution contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SCHEMA = "shohin-q36-mtr-graph-v1"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
TOTAL_ROWS = 1289
TRAIN_IDENTITIES = 5824
DRAFT_IDENTITIES = 7113
REVISION_PRESENTATIONS = 9655
SPLIT_SEED = 2026080811
DRAFT_SEED = 2026080818
REVISION_SEED = 2026080815
REVISION_DATA_SEED = 2026080814
EVALUATION_SEED = 2026080816
COMMIT_SEED = 2026080822
EXCLUDED_NODES = (
    "evc26",
    "evc29",
    "evc31",
    "evc32",
    "evc33",
    "evc37",
    "evc38",
    "evc46",
)
SOURCE_SHA256 = {
    "pairs": "45f1d66ce5e87dc2a1f4c3594bdde2bae26e9417e879d16eb4eddb228b696afe",
    "math": "e0ede83257e441050a019f59fb13d9c85bd6cba1d6a755ab86fb7129966ddbe5",
    "logic_science": "5a96859fd9088cde598b61da60dd2c6cb7281323ee06c034742a1b4e0e237017",
    "code": "0b6d068b4d71f407cb234579b9278dc640df09139ea906dd0f52a6ab71e05398",
    "b1": "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549",
}
PROHIBITED_RETRIES = (
    "ndr1",
    "kcr1",
    "vte1",
    "natural_language_microcode",
    "q35_edit_selector_cascade",
    "small_olmoe",
)
ARMS = (
    "learned_commit",
    "trained_revision",
    "unchanged",
    "self_refinement",
    "draft_hidden",
)
MIN_FREE_BYTES = 128 * 1024**3
MIN_FREE_INODES = 150_000
MAXIMUM_CONCURRENT_SINGLE_H100_REQUESTS = 32


class Q36MTRContractError(RuntimeError):
    """The prospective graph differs from its frozen contract."""


@dataclass(frozen=True)
class Stage:
    name: str
    h100_per_task: int
    tasks: int
    expected_h100_hours: float
    dependencies: tuple[str, ...]


STAGES = (
    Stage("preflight_cpu", 0, 1, 0.0, ()),
    Stage("mechanics", 1, 1, 0.25, ("preflight_cpu",)),
    Stage("owner_fit", 1, 1, 2.25, ("mechanics",)),
    Stage("draft_generate", 1, 16, 29.50, ("owner_fit",)),
    Stage("draft_merge", 0, 1, 0.0, ("draft_generate",)),
    Stage("materialize", 0, 1, 0.0, ("draft_merge",)),
    Stage("aligned_fit", 1, 1, 2.25, ("materialize",)),
    Stage("draft_hidden_fit", 1, 1, 2.25, ("materialize",)),
    Stage("calibration_revision", 1, 4, 1.30, ("aligned_fit",)),
    Stage("calibration_unchanged", 1, 4, 14.30, ("materialize",)),
    Stage("calibration_revision_merge", 0, 1, 0.0, ("calibration_revision",)),
    Stage("calibration_unchanged_merge", 0, 1, 0.0, ("calibration_unchanged",)),
    Stage(
        "commit_pairs",
        0,
        1,
        0.0,
        ("calibration_revision_merge", "calibration_unchanged_merge"),
    ),
    Stage("commit_fit", 1, 1, 0.50, ("commit_pairs",)),
    Stage("development_revision", 1, 8, 0.30, ("aligned_fit",)),
    Stage("development_unchanged", 1, 8, 3.24, ("materialize",)),
    Stage("development_self_refinement", 1, 8, 2.46, ("materialize",)),
    Stage("development_draft_hidden", 1, 8, 0.30, ("draft_hidden_fit",)),
    Stage("development_revision_merge", 0, 1, 0.0, ("development_revision",)),
    Stage("development_unchanged_merge", 0, 1, 0.0, ("development_unchanged",)),
    Stage(
        "development_self_refinement_merge",
        0,
        1,
        0.0,
        ("development_self_refinement",),
    ),
    Stage(
        "development_draft_hidden_merge",
        0,
        1,
        0.0,
        ("development_draft_hidden",),
    ),
    Stage(
        "commit_apply",
        0,
        1,
        0.0,
        ("commit_fit", "development_revision_merge", "development_unchanged_merge"),
    ),
    Stage(
        "precompute_custody",
        0,
        1,
        0.0,
        (
            "commit_apply",
            "development_self_refinement_merge",
            "development_draft_hidden_merge",
        ),
    ),
    Stage("prescore_accounting", 0, 1, 0.0, ("precompute_custody",)),
    Stage("authorize_score", 0, 1, 0.0, ("prescore_accounting",)),
    Stage("score_once", 0, 1, 0.0, ("authorize_score",)),
    Stage("normalize", 0, 1, 0.0, ("score_once",)),
    Stage("final_accounting", 0, 1, 0.0, ("normalize",)),
    Stage("compute_custody", 0, 1, 0.0, ("final_accounting",)),
    Stage("final_compare", 0, 1, 0.0, ("compute_custody",)),
)


def _stage_payloads() -> list[dict[str, Any]]:
    return [
        {**asdict(stage), "dependencies": list(stage.dependencies)} for stage in STAGES
    ]


def _hex_digest(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def graph_payload(source_commit: str) -> dict[str, Any]:
    if not _hex_digest(source_commit, 40):
        raise Q36MTRContractError("source commit must be one exact Git commit")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "prospective_no_submit",
        "source_commit": source_commit,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "scientific_submit_authorized": False,
        "model_acquisition_authorized": False,
        "data_materialization_authorized": False,
        "partition": "normal",
        "excluded_nodes": list(EXCLUDED_NODES),
        "requeue": False,
        "gpu_request": "gpu:nvidia_h100_pcie:1",
        "minimum_storage": {
            "free_bytes": MIN_FREE_BYTES,
            "free_inodes": MIN_FREE_INODES,
            "checked_after_model_runtime_install": True,
        },
        "data": {
            "source_sha256": dict(SOURCE_SHA256),
            "split_seed": SPLIT_SEED,
            "train_identities": TRAIN_IDENTITIES,
            "development_identities": TOTAL_ROWS,
            "draft_identities": DRAFT_IDENTITIES,
            "revision_presentations": REVISION_PRESENTATIONS,
            "source_disjoint": True,
            "holdout_access_authorized": False,
            "product_access_authorized": False,
            "public_access_authorized": False,
        },
        "training": {
            "owner_updates": 256,
            "revision_updates": 256,
            "draft_hidden_updates": 256,
            "commit_updates": 128,
            "revision_learning_rate": 2e-5,
            "controlled_layers": 16,
            "rank": 18,
            "alpha": 18,
            "draft_seed": DRAFT_SEED,
            "revision_seed": REVISION_SEED,
            "revision_data_seed": REVISION_DATA_SEED,
            "evaluation_seed": EVALUATION_SEED,
            "commit_seed": COMMIT_SEED,
            "max_sequence_length": 4096,
            "max_new_tokens": 768,
        },
        "arms": list(ARMS),
        "prohibited_retries": list(PROHIBITED_RETRIES),
        "stages": _stage_payloads(),
        "h100_requests": sum(stage.tasks * stage.h100_per_task for stage in STAGES),
        "expected_h100_hours": round(
            sum(stage.expected_h100_hours for stage in STAGES), 2
        ),
        # The four independent eight-shard development arms may all be ready
        # together. Concurrency is across stages, not merely within one array.
        "maximum_concurrent_single_h100_requests": (
            MAXIMUM_CONCURRENT_SINGLE_H100_REQUESTS
        ),
        "automatic_retry": False,
        "automatic_confirmation": False,
        "automatic_successor": False,
        "one_output_per_identity": True,
        "cancel_dead_dependencies_at_terminal": True,
        "temporary_shard_deletion_requires_verified_merge_and_mirror": True,
    }
    validate_graph(payload)
    return payload


def validate_graph(payload: dict[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("status") != (
        "prospective_no_submit"
    ):
        raise Q36MTRContractError("Q36-MTR graph schema/status differs")
    if payload.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise Q36MTRContractError("Q36-MTR host differs")
    if not _hex_digest(payload.get("source_commit"), 40):
        raise Q36MTRContractError("Q36-MTR source commit differs")
    for field in (
        "scientific_submit_authorized",
        "model_acquisition_authorized",
        "data_materialization_authorized",
        "requeue",
        "automatic_retry",
        "automatic_confirmation",
        "automatic_successor",
    ):
        if payload.get(field) is not False:
            raise Q36MTRContractError(f"Q36-MTR unsafe authorization: {field}")
    stages = payload.get("stages")
    if stages != _stage_payloads():
        raise Q36MTRContractError("Q36-MTR stages differ")
    seen: set[str] = set()
    for stage in stages:
        if not set(stage["dependencies"]) <= seen:
            raise Q36MTRContractError("Q36-MTR graph is not topologically ordered")
        seen.add(stage["name"])
    expected_requests = sum(stage.tasks * stage.h100_per_task for stage in STAGES)
    expected_hours = sum(stage.expected_h100_hours for stage in STAGES)
    if payload.get("h100_requests") != 61 or expected_requests != 61:
        raise Q36MTRContractError("Q36-MTR H100 request count differs")
    if (
        not isinstance(payload.get("expected_h100_hours"), (int, float))
        or not math.isclose(payload["expected_h100_hours"], 58.90, abs_tol=1e-12)
        or not math.isclose(expected_hours, 58.90, abs_tol=1e-12)
    ):
        raise Q36MTRContractError("Q36-MTR H100-hour plan differs")
    if payload.get("maximum_concurrent_single_h100_requests") != (
        MAXIMUM_CONCURRENT_SINGLE_H100_REQUESTS
    ):
        raise Q36MTRContractError("Q36-MTR maximum concurrency differs")
    if payload.get("excluded_nodes") != list(EXCLUDED_NODES):
        raise Q36MTRContractError("Q36-MTR scheduler exclusions differ")
    if payload.get("partition") != "normal" or payload.get("gpu_request") != (
        "gpu:nvidia_h100_pcie:1"
    ):
        raise Q36MTRContractError("Q36-MTR scheduler request differs")
    if payload.get("minimum_storage") != {
        "free_bytes": MIN_FREE_BYTES,
        "free_inodes": MIN_FREE_INODES,
        "checked_after_model_runtime_install": True,
    }:
        raise Q36MTRContractError("Q36-MTR storage gate differs")
    if payload.get("arms") != list(ARMS):
        raise Q36MTRContractError("Q36-MTR arm set differs")
    if payload.get("prohibited_retries") != list(PROHIBITED_RETRIES):
        raise Q36MTRContractError("Q36-MTR prohibited retries differ")
    if payload.get("data") != {
        "source_sha256": SOURCE_SHA256,
        "split_seed": SPLIT_SEED,
        "train_identities": TRAIN_IDENTITIES,
        "development_identities": TOTAL_ROWS,
        "draft_identities": DRAFT_IDENTITIES,
        "revision_presentations": REVISION_PRESENTATIONS,
        "source_disjoint": True,
        "holdout_access_authorized": False,
        "product_access_authorized": False,
        "public_access_authorized": False,
    }:
        raise Q36MTRContractError("Q36-MTR data contract differs")
    if payload.get("training") != {
        "owner_updates": 256,
        "revision_updates": 256,
        "draft_hidden_updates": 256,
        "commit_updates": 128,
        "revision_learning_rate": 2e-5,
        "controlled_layers": 16,
        "rank": 18,
        "alpha": 18,
        "draft_seed": DRAFT_SEED,
        "revision_seed": REVISION_SEED,
        "revision_data_seed": REVISION_DATA_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "commit_seed": COMMIT_SEED,
        "max_sequence_length": 4096,
        "max_new_tokens": 768,
    }:
        raise Q36MTRContractError("Q36-MTR training contract differs")
    for field in (
        "one_output_per_identity",
        "cancel_dead_dependencies_at_terminal",
        "temporary_shard_deletion_requires_verified_merge_and_mirror",
    ):
        if payload.get(field) is not True:
            raise Q36MTRContractError(f"Q36-MTR required invariant differs: {field}")
    if any(not _hex_digest(value) for value in SOURCE_SHA256.values()):
        raise Q36MTRContractError("Q36-MTR source hash is malformed")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRContractError(f"refusing existing Q36-MTR graph: {path}")
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
        raise Q36MTRContractError(f"refusing existing Q36-MTR graph: {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if (args.output is None) == (args.check is None):
        raise Q36MTRContractError("select exactly one of --output or --check")
    if args.check is not None:
        payload = json.loads(args.check.read_text(encoding="utf-8"))
        validate_graph(payload)
        if payload.get("source_commit") != args.source_commit:
            raise Q36MTRContractError("Q36-MTR graph source commit differs")
        print(hashlib.sha256(args.check.read_bytes()).hexdigest())
        return 0
    payload = graph_payload(args.source_commit)
    assert args.output is not None
    _atomic_json(args.output, payload)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
