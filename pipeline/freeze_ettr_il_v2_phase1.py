"""Freeze the no-fit ETTR-IL-v2 Phase-1 architecture handoff.

The freeze binds executable architecture and validation sources plus the two
deterministic CPU evidence artifacts.  It does not generate the production
population, load a checkpoint, instantiate a model, construct an optimizer,
fit weights, submit work, or open a scored split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Sequence


PROTOCOL = "R12-ETTR-IL-v2"
SCHEMA = "r12-ettr-il-v2-phase1-architecture-freeze-v1"
PROTECTED_CHECKPOINT_STEP = 300_000
PROTECTED_CHECKPOINT_SHA256 = (
    "211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6"
)
TOKENIZER_SHA256 = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)
TOKENIZER_PATH = "artifacts/shohin-tok-32k.json"
TOKENIZER_BYTES = 2_309_567
BASE_PARAMETERS = 125_081_664
ARCHITECTURE_PARAMETERS = 67_697_771
COMPLETE_PARAMETERS = 192_779_435
PARAMETER_CAP = 200_000_000

SPEC_PATHS = (
    "R12_ETTR_ISOLATED_LEARNABILITY_PREREG_V2.md",
    "R12_ETTR_IL_V2_SEMANTIC_GENERATOR_SPEC.md",
    "R12_ETTR_IL_V2_MATERIALIZATION_SPEC.md",
    "R12_ETTR_IL_V2_ARMS_AND_STATISTICS_SPEC.md",
    "R12_ETTR_IL_V2_CUSTODY_SPEC.md",
)
PIPELINE_SOURCE_PATHS = (
    "pipeline/audit_ettr_il_v2_specs.py",
    "pipeline/audit_ettr_il_v2_surface_capacity.py",
    "pipeline/ettr_il_v2_canary.py",
    "pipeline/ettr_il_v2_candidate_search.py",
    "pipeline/ettr_il_v2_controls.py",
    "pipeline/ettr_il_v2_custody.py",
    "pipeline/ettr_il_v2_dataset.py",
    "pipeline/ettr_il_v2_evaluator.py",
    "pipeline/ettr_il_v2_horn_adapter.py",
    "pipeline/ettr_il_v2_materialize.py",
    "pipeline/ettr_il_v2_resource_adapter.py",
    "pipeline/ettr_il_v2_rewrite_adapter.py",
    "pipeline/ettr_il_v2_schedule.py",
    "pipeline/ettr_il_v2_semantics.py",
    "pipeline/ettr_il_v2_statistics.py",
    "pipeline/ettr_il_v2_surface.py",
    "pipeline/ettr_il_v2_surface_adapter.py",
    "pipeline/ettr_il_v2_token_native_surface.py",
    "pipeline/freeze_ettr_il_v2_phase1.py",
)
TRAIN_SOURCE_PATHS = (
    "train/endogenous_typed_theory_reactor.py",
    "train/ettr_checkpoint.py",
    "train/ettr_data_contract.py",
    "train/ettr_episode.py",
    "train/ettr_il_v2_arms.py",
    "train/ettr_il_v2_readiness.py",
    "train/ettr_il_v2_source_deletion.py",
    "train/ettr_model_assembly.py",
    "train/ettr_objectives.py",
    "train/ettr_optimization.py",
    "train/ettr_train_step.py",
    "train/model.py",
    "train/muon.py",
)
TEST_SOURCE_PATHS = (
    "pipeline/test_audit_ettr_il_v2_specs.py",
    "pipeline/test_audit_ettr_il_v2_surface_capacity.py",
    "pipeline/test_ettr_il_v2_canary.py",
    "pipeline/test_ettr_il_v2_candidate_search.py",
    "pipeline/test_ettr_il_v2_controls.py",
    "pipeline/test_ettr_il_v2_custody.py",
    "pipeline/test_ettr_il_v2_dataset.py",
    "pipeline/test_ettr_il_v2_evaluator.py",
    "pipeline/test_ettr_il_v2_horn_adapter.py",
    "pipeline/test_ettr_il_v2_materialize.py",
    "pipeline/test_ettr_il_v2_resource_adapter.py",
    "pipeline/test_ettr_il_v2_rewrite_adapter.py",
    "pipeline/test_ettr_il_v2_schedule.py",
    "pipeline/test_ettr_il_v2_semantics.py",
    "pipeline/test_ettr_il_v2_statistics.py",
    "pipeline/test_ettr_il_v2_surface.py",
    "pipeline/test_ettr_il_v2_surface_adapter.py",
    "pipeline/test_ettr_il_v2_token_native_surface.py",
    "pipeline/test_freeze_ettr_il_v2_phase1.py",
    "train/test_ettr_il_v2_arms.py",
    "train/test_ettr_il_v2_readiness.py",
    "train/test_ettr_il_v2_source_deletion.py",
)
EVIDENCE_PATHS = (
    "artifacts/r12/ettr_il_v2_spec_integration_audit.json",
    "artifacts/r12/ettr_il_v2_surface_capacity_audit.json",
    "artifacts/r12/ettr_il_v2_end_to_end_canary.json",
)


class Phase1FreezeError(ValueError):
    """The architecture handoff is incomplete or internally inconsistent."""


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(root: Path, relative: str) -> bytes:
    path = root / relative
    try:
        before = path.lstat()
    except OSError as exc:
        raise Phase1FreezeError(f"required file is unavailable: {relative}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise Phase1FreezeError(
            f"required file is not a single-link regular file: {relative}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Phase1FreezeError(f"required file cannot be opened: {relative}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise Phase1FreezeError(f"required file changed before read: {relative}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise Phase1FreezeError(f"required file changed during read: {relative}")
        payload = b"".join(chunks)
        if len(payload) != opened.st_size:
            raise Phase1FreezeError(f"required file size changed: {relative}")
        return payload
    finally:
        os.close(descriptor)


def _strict_json(payload: bytes, relative: str) -> dict[str, Any]:
    if not payload.endswith(b"\n"):
        raise Phase1FreezeError(f"evidence is not canonical JSON: {relative}")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase1FreezeError(f"evidence is malformed: {relative}") from exc
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise Phase1FreezeError(f"evidence is not canonical JSON: {relative}")
    return value


def _inventory(root: Path, paths: Sequence[str]) -> list[dict[str, object]]:
    records = []
    for relative in sorted(paths):
        payload = _read_regular(root, relative)
        records.append(
            {
                "bytes": len(payload),
                "path": relative,
                "sha256": _sha256(payload),
            }
        )
    return records


def build_phase1_freeze(root: Path) -> dict[str, object]:
    root = root.resolve()
    tokenizer_payload = _read_regular(root, TOKENIZER_PATH)
    if (
        len(tokenizer_payload) != TOKENIZER_BYTES
        or _sha256(tokenizer_payload) != TOKENIZER_SHA256
    ):
        raise Phase1FreezeError("tokenizer identity differs")
    source_paths = (*SPEC_PATHS, *PIPELINE_SOURCE_PATHS, *TRAIN_SOURCE_PATHS)
    sources = _inventory(root, source_paths)
    tests = _inventory(root, TEST_SOURCE_PATHS)
    evidence = _inventory(root, EVIDENCE_PATHS)
    evidence_values = {
        relative: _strict_json(_read_regular(root, relative), relative)
        for relative in EVIDENCE_PATHS
    }
    integration = evidence_values[
        "artifacts/r12/ettr_il_v2_spec_integration_audit.json"
    ]
    capacity = evidence_values[
        "artifacts/r12/ettr_il_v2_surface_capacity_audit.json"
    ]
    canary = evidence_values[
        "artifacts/r12/ettr_il_v2_end_to_end_canary.json"
    ]
    canary_results = canary.get("results")
    canary_result_records = (
        canary_results
        if isinstance(canary_results, list)
        and all(type(value) is dict for value in canary_results)
        else []
    )
    if (
        integration.get("status") != "pass"
        or capacity.get("status") != "token_native_transport_capacity_pass"
        or capacity.get("fixed_transport_capacity_pass") is not True
        or capacity.get("tokenizer_sha256") != TOKENIZER_SHA256
        or canary.get("status") != "pass"
        or canary.get("tokenizer_sha256") != TOKENIZER_SHA256
        or len(canary_result_records) != 3
        or {value.get("ontology") for value in canary_result_records}
        != {"horn", "rewrite", "resource"}
        or any(
            value.get("row_count") != 16
            or value.get("causal_rectangle_count") != 4
            or value.get("source_free_batch") is not True
            or value.get("world_token_count") != 192
            or value.get("command_token_count") != 96
            or value.get("query_token_count") != 48
            for value in canary_result_records
        )
    ):
        raise Phase1FreezeError("Phase-1 evidence gates differ")
    if (
        BASE_PARAMETERS + ARCHITECTURE_PARAMETERS != COMPLETE_PARAMETERS
        or COMPLETE_PARAMETERS > PARAMETER_CAP
    ):
        raise Phase1FreezeError("parameter ledger differs")
    source_root = _sha256(canonical_json_bytes(sources))
    test_root = _sha256(canonical_json_bytes(tests))
    evidence_root = _sha256(canonical_json_bytes(evidence))
    return {
        "architecture": {
            "added_parameters": ARCHITECTURE_PARAMETERS,
            "base_parameters": BASE_PARAMETERS,
            "complete_parameters": COMPLETE_PARAMETERS,
            "headroom": PARAMETER_CAP - COMPLETE_PARAMETERS,
            "parameter_cap": PARAMETER_CAP,
            "reactor_horizon": 64,
            "relation_roles": 16,
            "state_slots": 64,
            "state_types": 8,
            "value_codes": 256,
        },
        "authorization": {
            "fitting_authorized": False,
            "phase1_complete": True,
            "phase2_authorized": False,
            "pretraining_authorized": False,
            "production_population_materialized": False,
            "reasoning_capability_claimed": False,
            "weight_updates_performed": 0,
        },
        "decision": (
            "r12_ettr_il_v2_phase1_architecture_frozen_"
            "phase2_requires_explicit_user_authorization"
        ),
        "evidence": evidence,
        "evidence_root_sha256": evidence_root,
        "phase2_entry_order": [
            "generate_and_certify_literal_population",
            "materialize_freeze_reload_and_replay_all_batches",
            "run_complete_leakage_and_metadata_audits",
            "validate_equal_budget_arms_and_zero_update_readiness",
            "request_explicit_user_authorization",
            "fit_only_after_authorization",
        ],
        "protected_checkpoint": {
            "read_by_freeze": False,
            "sha256": PROTECTED_CHECKPOINT_SHA256,
            "step": PROTECTED_CHECKPOINT_STEP,
        },
        "protocol": PROTOCOL,
        "schema": SCHEMA,
        "source_inventory": sources,
        "source_inventory_sha256": source_root,
        "status": "pass",
        "test_inventory": tests,
        "test_inventory_sha256": test_root,
        "tokenizer": {
            "bytes": TOKENIZER_BYTES,
            "path": TOKENIZER_PATH,
            "sha256": TOKENIZER_SHA256,
        },
    }


def publish_no_replace(report: dict[str, object], destination: Path) -> str:
    payload = canonical_json_bytes(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise Phase1FreezeError("short Phase-1 freeze write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _sha256(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_phase1_freeze(args.root)
    digest = publish_no_replace(report, args.output)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE_PARAMETERS",
    "COMPLETE_PARAMETERS",
    "PARAMETER_CAP",
    "PROTOCOL",
    "Phase1FreezeError",
    "SCHEMA",
    "build_phase1_freeze",
    "canonical_json_bytes",
    "publish_no_replace",
]
