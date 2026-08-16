#!/usr/bin/env python3
"""Create the sole safe PCF1 source/B1/model-manifest root from pinned inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any

from build_pcf1_data import freeze_sources
from pcf1_code_sandbox import (
    qualify_allocation,
    preflight_mbpp_reference,
    preflight_mbpp_setup,
)

B1_SHA256 = "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549"
MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
MODEL_ROOT = Path(
    "/lustre/fs1/home/sa305415/shohin/artifacts/external/"
    "ministral-3-8b-reasoning-2512-81eaece"
)
MODEL_MANIFEST_SHA256 = (
    "46cc9203a18a414e08a53109662c3802b57c046896185ca9ab31875e8167cf1f"
)
MODEL_CONFIG_SHA256 = "5aae04beb9f2a9949eb1df870cf47ba292012a066bdcdcb115a9ac43425f8086"
MODEL_SOURCE_REVISION_SHA256 = (
    "3576c1bfaa0652940d12817ad3267ffe65645dc558ceb9a153ffb72f7211a982"
)
MODEL_MANIFEST_FILES = 58
MODEL_MANIFEST_BYTES = 35_706_515_534
SCHEMA = "shohin-pcf1-prepare-receipt-v1"
CPU_RECEIPT_SCHEMA = "shohin-pcf1-prepare-custodian-receipt-v1"
PROTECTED = ("holdout", "product", "public")


class PCF1PrepareError(RuntimeError):
    """The narrow legacy-to-safe PCF1 preparation contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()


def copy_exact(source: Path, destination: Path, expected_sha256: str) -> None:
    if sha256_file(source) != expected_sha256:
        raise PCF1PrepareError("PCF1 safe-copy source hash differs")
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1 << 20)
        writer.flush()
        os.fsync(writer.fileno())
    if sha256_file(destination) != expected_sha256:
        raise PCF1PrepareError("PCF1 safe-copy destination hash differs")


def verify_model_manifest(
    model_root: Path,
    *,
    expected_sha256: str | None = None,
    expected_files: int | None = None,
    expected_bytes: int | None = None,
) -> tuple[int, int]:
    expected_sha256 = expected_sha256 or MODEL_MANIFEST_SHA256
    expected_files = expected_files or MODEL_MANIFEST_FILES
    expected_bytes = expected_bytes or MODEL_MANIFEST_BYTES
    manifest = model_root / "SHA256SUMS"
    if not manifest.is_file() or sha256_file(manifest) != expected_sha256:
        raise PCF1PrepareError("PCF1 authoritative model manifest differs")
    entries: list[tuple[str, Path]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        path = Path(relative)
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or path.is_absolute()
            or ".." in path.parts
            or not relative
            or any(term in relative.casefold() for term in PROTECTED)
        ):
            raise PCF1PrepareError("PCF1 authoritative model manifest entry differs")
        entries.append((digest, Path(path.as_posix())))
    relative_names = [path.as_posix() for _, path in entries]
    if (
        len(entries) != expected_files
        or relative_names != sorted(relative_names)
        or len(set(relative_names)) != len(relative_names)
    ):
        raise PCF1PrepareError("PCF1 authoritative model manifest geometry differs")
    covered_bytes = 0
    for expected, relative in entries:
        path = model_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise PCF1PrepareError("PCF1 authoritative model file differs")
        covered_bytes += path.stat().st_size
    actual = set()
    actual_directories = set()
    special = False
    for path in model_root.rglob("*"):
        relative = path.relative_to(model_root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            actual.add(relative)
        elif stat.S_ISDIR(mode):
            actual_directories.add(relative)
        else:
            special = True
    expected_directories = {
        parent.as_posix()
        for relative in (*relative_names, "SHA256SUMS")
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    if (
        actual != {*relative_names, "SHA256SUMS"}
        or actual_directories != expected_directories
        or special
        or covered_bytes != expected_bytes
    ):
        raise PCF1PrepareError("PCF1 authoritative model tree differs")
    return len(entries), covered_bytes


def validate_model_snapshot(
    model_root: Path, revision: str, *, expected_root: Path | None = None
) -> dict[str, Any]:
    resolved = model_root.resolve()
    if resolved != (expected_root or MODEL_ROOT).resolve(strict=False):
        raise PCF1PrepareError("PCF1 model cache repository/revision path differs")
    try:
        config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PCF1PrepareError("PCF1 model config is unreadable") from error
    text = config.get("text_config")
    vision = config.get("vision_config")
    if (
        config.get("model_type") != "mistral3"
        or config.get("architectures") != ["Mistral3ForConditionalGeneration"]
        or not isinstance(text, dict)
        or text.get("model_type") != "ministral3"
        or text.get("hidden_size") != 4096
        or text.get("num_hidden_layers") != 34
        or not isinstance(vision, dict)
        or vision.get("model_type") != "pixtral"
    ):
        raise PCF1PrepareError("PCF1 model config/layout differs")
    return config


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_revision != MODEL_REVISION:
        raise PCF1PrepareError("PCF1 model revision differs")
    if args.output.exists() or args.output.is_symlink():
        raise PCF1PrepareError(f"refusing existing PCF1 prepare root: {args.output}")
    protected_outputs = (
        args.output,
        args.assessor_output,
        args.assessor_receipt_output,
        args.cpu_receipt_output,
    )
    if any(
        term in str(path.resolve(strict=False)).casefold()
        for path in protected_outputs
        for term in PROTECTED
    ):
        raise PCF1PrepareError("PCF1 safe output path is protected")
    if any(path.exists() or path.is_symlink() for path in protected_outputs):
        raise PCF1PrepareError("refusing existing PCF1 prepare output")
    if any(
        path.resolve(strict=False).is_relative_to(args.output.resolve(strict=False))
        for path in (
            args.assessor_output,
            args.assessor_receipt_output,
            args.cpu_receipt_output,
        )
    ):
        raise PCF1PrepareError("PCF1 custodian output must be outside GPU-safe root")
    for path in (
        args.pairs,
        args.math_bank,
        args.science_bank,
        args.code_bank,
        args.b1_data,
        args.model_root / "config.json",
        args.model_root / "SOURCE_REVISION",
        args.model_root / "SHA256SUMS",
        args.environment_receipt,
    ):
        if not path.is_file():
            raise PCF1PrepareError(f"missing PCF1 prepare input: {path}")
    validate_model_snapshot(args.model_root, args.model_revision)
    if sha256_file(args.model_root / "config.json") != MODEL_CONFIG_SHA256:
        raise PCF1PrepareError("PCF1 authoritative model config hash differs")
    if sha256_file(args.model_root / "SOURCE_REVISION") != MODEL_SOURCE_REVISION_SHA256:
        raise PCF1PrepareError("PCF1 authoritative source revision hash differs")
    model_files, model_bytes = verify_model_manifest(args.model_root)
    reference_sandbox_receipt = qualify_allocation()
    setup_qualifications: dict[str, dict[str, Any]] = {}

    def evaluate_reference(source: dict[str, Any], split: str) -> dict[str, Any]:
        setup = str(source.get("test_setup_code") or "")
        setup_sha256 = hashlib.sha256(setup.encode()).hexdigest()
        if setup_sha256 not in setup_qualifications:
            setup_qualifications[setup_sha256] = preflight_mbpp_setup(setup)
        return preflight_mbpp_reference(
            source,
            split=split,
            setup_qualification=setup_qualifications[setup_sha256],
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise PCF1PrepareError("PCF1 prepare temporary root already exists")
    temporary.mkdir()
    assessor_published = False
    safe_root_published = False
    cpu_receipt_published = False
    try:
        sources = temporary / "sources"
        freeze = freeze_sources(
            pairs_path=args.pairs,
            bank_paths=(args.math_bank, args.science_bank, args.code_bank),
            output=sources,
            assessor_output=args.assessor_output,
            assessor_receipt_output=args.assessor_receipt_output,
            reference_evaluator=evaluate_reference,
            reference_sandbox_receipt=reference_sandbox_receipt,
        )
        if freeze.get("status") != "complete":
            raise PCF1PrepareError("PCF1 source freeze did not complete")
        assessor_published = True
        source_report_path = sources / "report.json"
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        assessor_receipt_record = source_report.get("outputs", {}).get(
            "confirmation_assessor_receipt"
        )
        assessor_receipt = json.loads(
            args.assessor_receipt_output.read_text(encoding="utf-8")
        )
        assessor_board_sha256 = (
            assessor_receipt_record.get("board_sha256")
            if isinstance(assessor_receipt_record, dict)
            else None
        )
        if (
            not isinstance(assessor_receipt_record, dict)
            or assessor_receipt_record.get("sha256")
            != sha256_file(args.assessor_receipt_output)
            or not isinstance(assessor_board_sha256, str)
            or len(assessor_board_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in assessor_board_sha256
            )
            or assessor_receipt.get("schema")
            != "shohin-pcf1-confirmation-assessor-receipt-v1"
            or assessor_receipt.get("status") != "complete"
            or set(assessor_receipt)
            != {
                "schema",
                "status",
                "board_sha256",
                "rows",
                "semantic_access",
            }
            or assessor_receipt.get("board_sha256") != assessor_board_sha256
            or assessor_receipt.get("rows") != 1289
            or assessor_receipt.get("semantic_access") != "final_score_only"
        ):
            raise PCF1PrepareError("PCF1 confirmation assessor receipt differs")
        b1 = temporary / "b1_train.jsonl"
        copy_exact(args.b1_data, b1, B1_SHA256)
        environment = temporary / "environment_receipt.json"
        environment_payload = json.loads(
            args.environment_receipt.read_text(encoding="utf-8")
        )
        if (
            environment_payload.get("schema") != "shohin-pcf1-environment-receipt-v1"
            or environment_payload.get("status") != "complete"
        ):
            raise PCF1PrepareError("PCF1 environment receipt differs")
        environment_sha256 = sha256_file(args.environment_receipt)
        copy_exact(args.environment_receipt, environment, environment_sha256)
        manifest = temporary / "model_manifest.sha256"
        copy_exact(args.model_root / "SHA256SUMS", manifest, MODEL_MANIFEST_SHA256)
        manifest_sha256 = MODEL_MANIFEST_SHA256
        config_sha256 = sha256_file(args.model_root / "config.json")
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "model_revision": args.model_revision,
            "legacy_inputs_path_recorded": False,
            "legacy_inputs_hash_pinned": True,
            "safe_paths_only": True,
            "outputs": {
                "sources": {
                    "path": str((args.output / "sources").resolve()),
                    "report_sha256": sha256_file(sources / "report.json"),
                },
                "b1": {
                    "path": str((args.output / "b1_train.jsonl").resolve()),
                    "sha256": B1_SHA256,
                },
                "model_manifest": {
                    "path": str((args.output / "model_manifest.sha256").resolve()),
                    "sha256": manifest_sha256,
                    "files": model_files,
                    "dereferenced_bytes": model_bytes,
                    "sorted_relative_paths": True,
                    "authoritative_manifest_copy": True,
                    "source_completion_job": "747023",
                },
                "environment": {
                    "path": str((args.output / "environment_receipt.json").resolve()),
                    "sha256": environment_sha256,
                    "schema": "shohin-pcf1-environment-receipt-v1",
                },
            },
            "model_root": str(args.model_root.resolve()),
            "model_config_sha256": config_sha256,
            "b1_control_plane_exception": {
                "input_sha256": B1_SHA256,
                "byte_transform_applied": False,
                "schema_field_visible_to_model": False,
                "evaluation_or_candidate_schema": False,
                "qualified_lineage_preserved": True,
            },
            "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        }
        atomic_json(temporary / "receipt.json", receipt)
        cpu_receipt = {
            "schema": CPU_RECEIPT_SCHEMA,
            "status": "complete",
            "gpu_safe_root": str(args.output.resolve()),
            "gpu_safe_receipt": str((args.output / "receipt.json").resolve()),
            "gpu_safe_receipt_sha256": sha256_file(temporary / "receipt.json"),
            "source_freeze_report": str(
                (args.output / "sources/report.json").resolve()
            ),
            "source_freeze_report_sha256": sha256_file(source_report_path),
            "confirmation_assessors": {
                "path": str(args.assessor_output.resolve()),
                "sha256": assessor_board_sha256,
                "rows": 1289,
                "semantic_access": "final_score_only",
                "gpu_exported": False,
            },
            "confirmation_assessor_receipt": {
                "path": str(args.assessor_receipt_output.resolve()),
                "sha256": sha256_file(args.assessor_receipt_output),
            },
            "sealed_access_authorized": False,
        }
        cpu_temporary = args.cpu_receipt_output.with_name(
            f".{args.cpu_receipt_output.name}.tmp.{os.getpid()}"
        )
        args.cpu_receipt_output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(cpu_temporary, cpu_receipt)
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temporary, args.output)
        safe_root_published = True
        try:
            os.link(cpu_temporary, args.cpu_receipt_output)
            cpu_receipt_published = True
        except FileExistsError as error:
            raise PCF1PrepareError(
                "refusing existing PCF1 custodian receipt"
            ) from error
        finally:
            cpu_temporary.unlink(missing_ok=True)
        for parent in {
            args.output.parent,
            args.assessor_output.parent,
            args.assessor_receipt_output.parent,
            args.cpu_receipt_output.parent,
        }:
            parent_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        if safe_root_published and args.output.exists():
            shutil.rmtree(args.output)
        if cpu_receipt_published:
            args.cpu_receipt_output.unlink(missing_ok=True)
        if assessor_published:
            args.assessor_receipt_output.unlink(missing_ok=True)
            args.assessor_output.unlink(missing_ok=True)
        raise
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--math-bank", type=Path, required=True)
    parser.add_argument("--science-bank", type=Path, required=True)
    parser.add_argument("--code-bank", type=Path, required=True)
    parser.add_argument("--b1-data", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--assessor-output", type=Path, required=True)
    parser.add_argument("--assessor-receipt-output", type=Path, required=True)
    parser.add_argument("--cpu-receipt-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    receipt = prepare(parser.parse_args())
    print(json.dumps({"status": receipt["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
