#!/usr/bin/env python3
"""Compose hash-bound ETTR components into an existing joint model.

This is an initialization/transplant operation, not an optimizer update.
The base transformer is copied byte-for-byte from the parent joint model.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Mapping, Sequence

import torch

from eval_ettr_v3 import _parameter_sha256, _read_hash_bound_json
from train_ettr_component_island import (
    _component_state,
    load_component_warm_start,
)
from train_ettr_joint_instruction_canary import (
    MODEL_SCHEMA,
    RUN_SCHEMA,
    _load_parent,
)


REPORT_SCHEMA = "shohin-ettr-joint-component-composition-report-v1"
COMPOSITION_KIND = "hash-bound-component-transplant"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENTS = ("compiler", "reactor", "reader")


class ETTRJointCompositionError(RuntimeError):
    """A joint-model composition custody contract failed."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_no_replace(path: Path, payload: bytes) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o400)
    except OSError as exc:
        raise ETTRJointCompositionError(
            "refusing an existing or unsafe composition output"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _torch_save_no_replace(
    path: Path,
    payload: Mapping[str, object],
) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o400)
    except OSError as exc:
        raise ETTRJointCompositionError(
            "refusing an existing or unsafe model output"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            torch.save(dict(payload), output)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _sha256_file(path)


def _composition_receipt(
    *,
    parent_joint_model: Path,
    parent_joint_model_sha256: str,
    parent_run_contract: Path,
    parent_run_contract_sha256: str,
    components: Mapping[str, Mapping[str, str]],
    source_commit: str,
) -> dict[str, object]:
    return {
        "components": {
            name: dict(components[name]) for name in _COMPONENTS
        },
        "kind": COMPOSITION_KIND,
        "optimizer_updates": 0,
        "parent_joint_model": str(parent_joint_model),
        "parent_joint_model_sha256": parent_joint_model_sha256,
        "parent_run_contract": str(parent_run_contract),
        "parent_run_contract_sha256": parent_run_contract_sha256,
        "source_commit": source_commit,
    }


def _composed_run_contract(
    parent_contract: Mapping[str, object],
    *,
    composition: Mapping[str, object],
    source_commit: str,
) -> dict[str, object]:
    contract = deepcopy(dict(parent_contract))
    if contract.get("schema") != RUN_SCHEMA:
        raise ETTRJointCompositionError(
            "parent tri-stream run contract schema differs"
        )
    if "component_composition" in contract:
        raise ETTRJointCompositionError(
            "refusing to recursively compose a component transplant"
        )
    contract["component_composition"] = deepcopy(dict(composition))
    contract["source_commit"] = source_commit
    return contract


def _component_snapshots(
    model: torch.nn.Module,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            key: value.clone()
            for key, value in _component_state(model, name).items()
        }
        for name in _COMPONENTS
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-joint-model", type=Path, required=True)
    parser.add_argument("--parent-joint-model-sha256", required=True)
    parser.add_argument("--parent-run-contract", type=Path, required=True)
    parser.add_argument("--parent-run-contract-sha256", required=True)
    for component in _COMPONENTS:
        parser.add_argument(
            f"--{component}",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{component}-sha256",
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    paths = (
        args.parent_joint_model,
        args.parent_run_contract,
        args.compiler,
        args.reactor,
        args.reader,
        args.output,
    )
    hashes = (
        args.parent_joint_model_sha256,
        args.parent_run_contract_sha256,
        args.compiler_sha256,
        args.reactor_sha256,
        args.reader_sha256,
    )
    if (
        any(not path.is_absolute() for path in paths)
        or any(_HEX64.fullmatch(value) is None for value in hashes)
        or _HEX40.fullmatch(args.source_commit) is None
        or args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise ETTRJointCompositionError(
            "joint component composition arguments differ"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    parent_contract = _read_hash_bound_json(
        args.parent_run_contract,
        expected_sha256=args.parent_run_contract_sha256,
        label="parent tri-stream run contract",
    )
    model, parent_payload = _load_parent(
        args.parent_joint_model,
        expected_sha256=args.parent_joint_model_sha256,
    )
    if (
        parent_payload.get("schema") != MODEL_SCHEMA
        or parent_payload.get("run_contract_sha256")
        != args.parent_run_contract_sha256
    ):
        raise ETTRJointCompositionError(
            "parent joint-model lineage differs"
        )

    component_paths = {
        name: getattr(args, name) for name in _COMPONENTS
    }
    component_hashes = {
        name: getattr(args, f"{name}_sha256") for name in _COMPONENTS
    }
    parent_parameter_sha256 = _parameter_sha256(model)
    parent_components = _component_snapshots(model)
    for name in _COMPONENTS:
        load_component_warm_start(
            model,
            name,
            component_paths[name],
            expected_sha256=component_hashes[name],
        )
    changed_tensors = {
        name: sum(
            not torch.equal(parent_components[name][key], value)
            for key, value in _component_state(model, name).items()
        )
        for name in _COMPONENTS
    }
    if not any(changed_tensors.values()):
        raise ETTRJointCompositionError(
            "component transplant did not change any component"
        )

    components = {
        name: {
            "path": str(component_paths[name]),
            "sha256": component_hashes[name],
        }
        for name in _COMPONENTS
    }
    composition = _composition_receipt(
        parent_joint_model=args.parent_joint_model,
        parent_joint_model_sha256=args.parent_joint_model_sha256,
        parent_run_contract=args.parent_run_contract,
        parent_run_contract_sha256=args.parent_run_contract_sha256,
        components=components,
        source_commit=args.source_commit,
    )
    run_contract = _composed_run_contract(
        parent_contract,
        composition=composition,
        source_commit=args.source_commit,
    )
    run_contract_bytes = _canonical_bytes(run_contract)
    run_contract_sha256 = hashlib.sha256(run_contract_bytes).hexdigest()
    model_payload = {
        "base_config": parent_payload["base_config"],
        "ettr_config": parent_payload["ettr_config"],
        "initialization": composition,
        "model": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        },
        "optimizer_step": parent_payload["optimizer_step"],
        "run_contract_sha256": run_contract_sha256,
        "schedule": parent_payload["schedule"],
        "schema": MODEL_SCHEMA,
        "source_commit": args.source_commit,
    }

    try:
        args.output.mkdir(mode=0o700)
        observed_contract_sha256 = _write_no_replace(
            args.output / "run-contract.json",
            run_contract_bytes,
        )
        if observed_contract_sha256 != run_contract_sha256:
            raise ETTRJointCompositionError(
                "composed run-contract hash differs"
            )
        joint_model_sha256 = _torch_save_no_replace(
            args.output / "joint-model-final.pt",
            model_payload,
        )
        composed_parameter_sha256 = _parameter_sha256(model)
        report = {
            "changed_component_tensors": changed_tensors,
            "component_composition": composition,
            "composed_parameter_sha256": composed_parameter_sha256,
            "joint_model_sha256": joint_model_sha256,
            "parent_parameter_sha256": parent_parameter_sha256,
            "run_contract_sha256": run_contract_sha256,
            "schema": REPORT_SCHEMA,
            "source_commit": args.source_commit,
        }
        _write_no_replace(
            args.output / "composition-report.json",
            _canonical_bytes(report),
        )
    except BaseException:
        shutil.rmtree(args.output, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "joint_model_sha256": joint_model_sha256,
                "output": str(args.output),
                "run_contract_sha256": run_contract_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
