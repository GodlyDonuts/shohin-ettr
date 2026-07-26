#!/usr/bin/env python3
"""Prepare, execute, and admit one synthetic ETTR supervisor smoke.

The ``prepare`` and ``validate`` phases use the production board, model,
tokenization, state, and custody APIs.  The ``run-chain`` phase is deliberately
stdlib-only so a root-owned host Python can execute it under a pinned source
hash.  It calls the production verified-archive extraction primitive once,
retains the extracted runtime by directory descriptor, and invokes the exact
supervisor shipped in that runtime for WORLD, COMMAND, and QUERY.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import runpy
import stat
import subprocess
import sys
from typing import Mapping, Sequence


FIXTURE_SCHEMA = "ettr-supervisor-synthetic-fixture-v1"
PLAN_SCHEMA = "ettr-supervisor-smoke-plan-v1"
CHAIN_SCHEMA = "ettr-supervisor-smoke-chain-v1"
REPORT_SCHEMA = "ettr-supervisor-smoke-report-v1"
CHECKPOINT_STEP = 0
MODEL_SEED = 2026072601
STAGES = ("world", "command", "query")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_32_BYTES = re.compile(r"^[0-9a-f]{64}$")
_SUPERVISOR_LOADER_CODE = (
    "import runpy,sys;"
    "root=sys.argv.pop(1);"
    "tools=root+'/app/tools';"
    "sys.path.insert(0,tools);"
    "runpy.run_path("
    "tools+'/ettr_stage_supervisor.py',run_name='__main__')"
)
_PUBLIC_KEY_LOADER_CODE = (
    "import os,sys;"
    "root=sys.argv[1];fd=int(sys.argv[2]);"
    "sys.path.insert(0,root+'/app/tools');"
    "from ettr_deployment_contract import "
    "ed25519_public_key_from_private_bytes as p;"
    "os.lseek(fd,0,0);"
    "key=os.read(fd,33);"
    "os.lseek(fd,0,0);"
    "sys.stdout.write(p(key).hex()+'\\n')"
)


class ETTRSupervisorSmokeError(RuntimeError):
    """The synthetic supervisor smoke differed from its exact contract."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_immutable(path: Path, *, max_bytes: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ETTRSupervisorSmokeError(
            f"immutable input cannot be opened: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o222
            or (
                max_bytes is not None
                and not 0 <= before.st_size <= max_bytes
            )
        ):
            raise ETTRSupervisorSmokeError(
                f"input is not immutable single-link: {path}"
            )
        payload = bytearray()
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            payload.extend(chunk)
        after = os.fstat(descriptor)
        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
        if identity(before) != identity(after) or len(payload) != before.st_size:
            raise ETTRSupervisorSmokeError(
                f"immutable input changed during read: {path}"
            )
        return bytes(payload)
    finally:
        os.close(descriptor)


def _read_pinned_source(
    path: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> bytes:
    """Hash stable source bytes before copying them into a sealed memfd."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ETTRSupervisorSmokeError(
            f"pinned source cannot be opened: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= max_bytes
        ):
            raise ETTRSupervisorSmokeError(
                f"pinned source geometry differs: {path}"
            )
        payload = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            identity_before != identity_after
            or len(payload) != before.st_size
            or _sha256_bytes(payload) != expected_sha256
        ):
            raise ETTRSupervisorSmokeError(
                f"pinned source identity differs: {path}"
            )
        return bytes(payload)
    finally:
        os.close(descriptor)


def _read_canonical_json(
    path: Path,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = 1024 * 1024,
) -> dict[str, object]:
    payload = _read_immutable(path, max_bytes=max_bytes)
    if (
        expected_sha256 is not None
        and _sha256_bytes(payload) != expected_sha256
    ):
        raise ETTRSupervisorSmokeError(f"JSON identity differs: {path}")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRSupervisorSmokeError(f"JSON is malformed: {path}") from exc
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value):
        raise ETTRSupervisorSmokeError(f"JSON is not canonical: {path}")
    return value


def _write_once(path: Path, payload: bytes, *, mode: int = 0o444) -> str:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise ETTRSupervisorSmokeError(
            f"refusing smoke output path: {path}"
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ETTRSupervisorSmokeError("smoke output write stalled")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    return _sha256_bytes(payload)


def _write_json_once(path: Path, value: object) -> str:
    return _write_once(path, _canonical_json_bytes(value))


def _copy_once(source: Path, destination: Path) -> str:
    return _write_once(destination, _read_immutable(source))


def _sha256_file(
    path: Path,
    *,
    root_owned_executable: bool = False,
) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ETTRSupervisorSmokeError(
            f"hashed input cannot be opened: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        ordinary_immutable = not (before.st_mode & 0o222)
        trusted_system_executable = (
            root_owned_executable
            and before.st_uid == 0
            and not (before.st_mode & 0o022)
            and bool(before.st_mode & 0o111)
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not (ordinary_immutable or trusted_system_executable)
        ):
            raise ETTRSupervisorSmokeError(
                f"hashed input is not immutable single-link: {path}"
            )
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or size != before.st_size:
            raise ETTRSupervisorSmokeError(
                f"hashed input changed during read: {path}"
            )
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _install_production_paths(source_root: Path) -> None:
    root = source_root.resolve(strict=True)
    for path in (root / "train", root / "pipeline"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


@dataclass(frozen=True, slots=True)
class SmokeRuntimeBindings:
    runtime_archive_path: Path
    claim_runtime_verification_receipt_path: Path
    runtime_receipt_paths: tuple[tuple[str, Path], ...]
    archive_sha256: str
    archive_size: int
    inventory_sha256: str
    source_bundle_sha256: str
    python_sha256: str
    bootstrap_sha256: str
    external_launcher_sha256: str
    bwrap_path: Path
    bwrap_sha256: str
    runtime_bundle_sha256s: tuple[tuple[str, str], ...]
    runner_sha256s: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        hashes = (
            self.archive_sha256,
            self.inventory_sha256,
            self.source_bundle_sha256,
            self.python_sha256,
            self.bootstrap_sha256,
            self.external_launcher_sha256,
            self.bwrap_sha256,
            *(value for _, value in self.runtime_bundle_sha256s),
            *(value for _, value in self.runner_sha256s),
        )
        if (
            any(_SHA256.fullmatch(value) is None for value in hashes)
            or self.archive_size <= 0
            or tuple(stage for stage, _ in self.runtime_receipt_paths)
            != STAGES
            or tuple(stage for stage, _ in self.runtime_bundle_sha256s)
            != STAGES
            or tuple(stage for stage, _ in self.runner_sha256s)
            != STAGES
            or not self.runtime_archive_path.is_absolute()
            or not self.bwrap_path.is_absolute()
        ):
            raise ETTRSupervisorSmokeError("runtime bindings differ")

    def runtime_receipt_path(self, stage: str) -> Path:
        return dict(self.runtime_receipt_paths)[stage]

    def runtime_bundle_sha256(self, stage: str) -> str:
        return dict(self.runtime_bundle_sha256s)[stage]

    def runner_sha256(self, stage: str) -> str:
        return dict(self.runner_sha256s)[stage]


def load_runtime_bindings(
    *,
    source_root: Path,
    runtime_archive_path: Path,
    runtime_inventory_path: Path,
    claim_runtime_verification_receipt_path: Path,
    runtime_receipt_paths: Mapping[str, Path],
    bwrap_path: Path,
    expected_archive_sha256: str,
    expected_inventory_sha256: str,
    expected_source_bundle_sha256: str,
    expected_bwrap_sha256: str,
) -> SmokeRuntimeBindings:
    """Load exact runtime sidecars through their production parsers."""

    _install_production_paths(source_root)
    from ettr_claim_runtime import (  # noqa: PLC0415
        ETTRClaimRuntimeInventory,
        ETTRClaimRuntimeVerificationReceipt,
    )
    from ettr_runtime_bundle import (  # noqa: PLC0415
        ETTRRuntimeBundleReceipt,
    )

    for digest in (
        expected_archive_sha256,
        expected_inventory_sha256,
        expected_source_bundle_sha256,
        expected_bwrap_sha256,
    ):
        if _SHA256.fullmatch(digest) is None:
            raise ETTRSupervisorSmokeError("runtime pin is not a SHA-256")
    inventory_payload = _read_immutable(
        runtime_inventory_path,
        max_bytes=64 * 1024 * 1024,
    )
    inventory = ETTRClaimRuntimeInventory.from_bytes(inventory_payload)
    if (
        inventory.sha256() != expected_inventory_sha256
        or inventory.source_bundle_sha256()
        != expected_source_bundle_sha256
    ):
        raise ETTRSupervisorSmokeError("runtime inventory pin differs")
    verification_payload = _read_canonical_json(
        claim_runtime_verification_receipt_path,
        max_bytes=64 * 1024,
    )
    try:
        verification_receipt = ETTRClaimRuntimeVerificationReceipt(
            **verification_payload
        )
    except TypeError as exc:
        raise ETTRSupervisorSmokeError(
            "runtime verification receipt fields differ"
        ) from exc
    verification_receipt.validate()
    archive_sha256, archive_size = _sha256_file(runtime_archive_path)
    bwrap_sha256, _ = _sha256_file(
        bwrap_path,
        root_owned_executable=True,
    )
    if (
        archive_sha256 != expected_archive_sha256
        or archive_size != verification_receipt.archive_size
        or verification_receipt.archive_sha256 != archive_sha256
        or verification_receipt.inventory_sha256
        != expected_inventory_sha256
        or bwrap_sha256 != expected_bwrap_sha256
    ):
        raise ETTRSupervisorSmokeError("runtime archive or loader pin differs")
    members = {member.path: member for member in inventory.members}

    def member_hash(path: str) -> str:
        try:
            value = members[path].sha256
        except KeyError as exc:
            raise ETTRSupervisorSmokeError(
                f"runtime member is absent: {path}"
            ) from exc
        if value is None:
            raise ETTRSupervisorSmokeError(
                f"runtime member is not a file: {path}"
            )
        return value

    parsed_runtime_receipts = []
    for stage in STAGES:
        try:
            path = runtime_receipt_paths[stage]
        except KeyError as exc:
            raise ETTRSupervisorSmokeError(
                "runtime receipt inventory differs"
            ) from exc
        receipt = ETTRRuntimeBundleReceipt.from_path(path)
        if receipt.stage != stage:
            raise ETTRSupervisorSmokeError("runtime receipt stage differs")
        parsed_runtime_receipts.append((stage, receipt.sha256()))
    bindings = SmokeRuntimeBindings(
        runtime_archive_path=runtime_archive_path.resolve(strict=True),
        claim_runtime_verification_receipt_path=(
            claim_runtime_verification_receipt_path.resolve(strict=True)
        ),
        runtime_receipt_paths=tuple(
            (stage, runtime_receipt_paths[stage].resolve(strict=True))
            for stage in STAGES
        ),
        archive_sha256=archive_sha256,
        archive_size=archive_size,
        inventory_sha256=inventory.sha256(),
        source_bundle_sha256=inventory.source_bundle_sha256(),
        python_sha256=verification_receipt.python_sha256,
        bootstrap_sha256=verification_receipt.bootstrap_sha256,
        external_launcher_sha256=member_hash(
            "app/tools/ettr_stage_supervisor.py"
        ),
        bwrap_path=bwrap_path.resolve(strict=True),
        bwrap_sha256=bwrap_sha256,
        runtime_bundle_sha256s=tuple(parsed_runtime_receipts),
        runner_sha256s=tuple(
            (
                stage,
                member_hash(
                    "app/candidate/"
                    f"{stage}/"
                    + {
                        "world": "run_ettr_world_compiler.py",
                        "command": "run_ettr_state_executor.py",
                        "query": "run_ettr_late_query.py",
                    }[stage]
                ),
            )
            for stage in STAGES
        ),
    )
    bindings.validate()
    return bindings


def _synthetic_model() -> object:
    import torch  # noqa: PLC0415
    from endogenous_typed_theory_reactor import (  # noqa: PLC0415
        EndogenousTypedTheoryReactorGPT,
        TheoryReactorConfig,
    )
    from model import GPT, GPTConfig  # noqa: PLC0415

    torch.manual_seed(MODEL_SEED)
    base = GPT(
        GPTConfig(
            vocab_size=256,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=32,
            d_ff=64,
            seq_len=512,
            zloss=0.0,
        )
    )
    return EndogenousTypedTheoryReactorGPT(
        base,
        TheoryReactorConfig(
            d_model=32,
            state_width=32,
            num_slots=6,
            num_types=3,
            num_relations=3,
            num_value_codes=64,
            max_edges=96,
            num_heads=4,
            compiler_layers=1,
            reactor_layers=1,
            query_layers=1,
            ff_multiplier=2,
            max_steps=6,
            stage_after_block=1,
            parameter_cap=1_000_000,
        ),
    ).eval()


def _write_character_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer  # noqa: PLC0415
    from tokenizers.models import WordLevel  # noqa: PLC0415
    from tokenizers.pre_tokenizers import Split  # noqa: PLC0415

    vocabulary = {"<pad>": 0, "<unk>": 1}
    vocabulary.update({chr(code): code + 2 for code in range(128)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Split("", behavior="isolated")
    temporary = path.with_name(f".{path.name}.temporary")
    tokenizer.save(str(temporary))
    _write_once(path, temporary.read_bytes())
    temporary.unlink()


def _save_component(path: Path, module: object) -> None:
    from safetensors.torch import save  # noqa: PLC0415

    payload = save(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in module.state_dict().items()
        }
    )
    _write_once(path, payload)


def prepare_fixture(
    *,
    source_root: Path,
    output_root: Path,
    runtime: SmokeRuntimeBindings,
) -> dict[str, object]:
    """Build one deterministic untrained model and production custody plan."""

    runtime.validate()
    _install_production_paths(source_root)
    import torch  # noqa: PLC0415
    from ettr_deployment_contract import (  # noqa: PLC0415
        canonical_stage_policy_sha256s,
    )
    from ettr_factorial_custody import (  # noqa: PLC0415
        ETTRFactorialExecutionManifest,
        EXECUTION_MANIFEST_SCHEMA,
    )
    from ettr_factorial_qualification_board import (  # noqa: PLC0415
        TOTAL_PACKETS,
        build_ettr_factorial_qualification_board,
    )
    from ettr_factorial_signed_custody import (  # noqa: PLC0415
        validate_primary_custody_receipts,
    )
    from ettr_factorial_tokenization import (  # noqa: PLC0415
        build_ettr_factorial_tokenization_receipt,
    )
    from ettr_model_assembly import ETTRModelAssemblyReceipt  # noqa: PLC0415

    if output_root.exists() or output_root.is_symlink():
        raise ETTRSupervisorSmokeError("fixture output already exists")
    output_root.mkdir(mode=0o700, parents=False)
    inputs = output_root / "inputs"
    inputs.mkdir(mode=0o700)
    model = _synthetic_model()
    board = build_ettr_factorial_qualification_board()

    tokenizer_path = inputs / "tokenizer.json"
    config_path = inputs / "config.json"
    checkpoint_path = inputs / "synthetic-base.pt"
    compiler_path = inputs / "compiler.safetensors"
    reactor_path = inputs / "reactor.safetensors"
    reader_path = inputs / "query-reader.safetensors"
    world_path = inputs / "world.json"
    command_path = inputs / "command.json"
    query_path = inputs / "query.json"
    tokenization_path = inputs / "tokenization-receipt.json"
    assembly_path = inputs / "model-assembly-receipt.json"
    manifest_path = inputs / "execution-manifest.json"
    claim_receipt_path = inputs / "claim-runtime-verification-receipt.json"
    runtime_receipt_outputs = {
        stage: inputs / f"runtime-receipt-{stage}.json"
        for stage in STAGES
    }

    _write_character_tokenizer(tokenizer_path)
    _write_json_once(config_path, asdict(model.config))
    checkpoint_payload = {
        "cfg": asdict(model.base.cfg),
        "model": model.base.state_dict(),
        "step": CHECKPOINT_STEP,
    }
    checkpoint_buffer = BytesIO()
    torch.save(
        checkpoint_payload,
        checkpoint_buffer,
        _use_new_zipfile_serialization=True,
    )
    _write_once(checkpoint_path, checkpoint_buffer.getvalue())
    _save_component(compiler_path, model.compiler)
    _save_component(reactor_path, model.reactor)
    _save_component(reader_path, model.query_reader)

    tokenization = build_ettr_factorial_tokenization_receipt(
        board,
        tokenizer_path,
        seq_len=model.base.cfg.seq_len,
        pad_token_id=0,
    )
    _write_once(tokenization_path, tokenization.canonical_bytes())
    _write_once(world_path, tokenization.stage_payload_bytes("world"))
    _write_once(command_path, tokenization.stage_payload_bytes("command"))
    _write_once(query_path, tokenization.stage_payload_bytes("query"))
    assembly = ETTRModelAssemblyReceipt.build(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_step=CHECKPOINT_STEP,
        compiler_path=compiler_path,
        reactor_path=reactor_path,
        query_reader_path=reader_path,
    )
    _write_once(assembly_path, assembly.canonical_bytes())
    _copy_once(
        runtime.claim_runtime_verification_receipt_path,
        claim_receipt_path,
    )
    for stage in STAGES:
        _copy_once(
            runtime.runtime_receipt_path(stage),
            runtime_receipt_outputs[stage],
        )

    policies = canonical_stage_policy_sha256s()
    manifest = ETTRFactorialExecutionManifest(
        schema=EXECUTION_MANIFEST_SCHEMA,
        board_sha256=board.receipt.payload_sha256,
        model_sha256=assembly.complete_model_sha256,
        config_sha256=assembly.config_sha256,
        checkpoint_sha256=assembly.checkpoint_sha256,
        checkpoint_step=CHECKPOINT_STEP,
        compiler_sha256=assembly.compiler_sha256,
        reactor_sha256=assembly.reactor_sha256,
        reader_sha256=assembly.query_reader_sha256,
        tokenizer_sha256=tokenization.tokenizer_sha256,
        tokenization_receipt_sha256=tokenization.sha256(),
        model_assembly_receipt_sha256=assembly.sha256(),
        bootstrap_sha256=runtime.bootstrap_sha256,
        world_runtime_bundle_sha256=runtime.runtime_bundle_sha256("world"),
        command_runtime_bundle_sha256=runtime.runtime_bundle_sha256("command"),
        query_runtime_bundle_sha256=runtime.runtime_bundle_sha256("query"),
        claim_runtime_archive_sha256=runtime.archive_sha256,
        claim_runtime_archive_size=runtime.archive_size,
        claim_runtime_inventory_sha256=runtime.inventory_sha256,
        external_launcher_sha256=runtime.external_launcher_sha256,
        bwrap_sha256=runtime.bwrap_sha256,
        network_namespace_required=True,
        world_stage_policy_sha256=policies["world"],
        command_stage_policy_sha256=policies["command"],
        query_stage_policy_sha256=policies["query"],
        compiler_runner_sha256=runtime.runner_sha256("world"),
        executor_runner_sha256=runtime.runner_sha256("command"),
        query_runner_sha256=runtime.runner_sha256("query"),
        compiler_hard=True,
        executor_hard=True,
        executor_steps=2,
        world_package_sha256=board.receipt.world_package_sha256,
        command_package_sha256=board.receipt.command_package_sha256,
        query_package_sha256=board.receipt.query_package_sha256,
        world_tokens_sha256=_sha256_file(world_path)[0],
        command_tokens_sha256=_sha256_file(command_path)[0],
        query_tokens_sha256=_sha256_file(query_path)[0],
        row_count=TOTAL_PACKETS,
    )
    manifest_sha256 = _write_json_once(manifest_path, asdict(manifest))
    validate_primary_custody_receipts(
        board,
        execution_manifest=manifest,
        expected_execution_manifest_sha256=manifest_sha256,
        tokenization_receipt=tokenization,
        tokenizer_path=tokenizer_path,
        model_assembly_receipt=assembly,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        compiler_path=compiler_path,
        reactor_path=reactor_path,
        query_reader_path=reader_path,
    )
    plan = {
        "schema": PLAN_SCHEMA,
        "fixture_schema": FIXTURE_SCHEMA,
        "checkpoint_kind": "deterministic-synthetic-untrained",
        "checkpoint_step": CHECKPOINT_STEP,
        "model_seed": MODEL_SEED,
        "runtime_archive_path": str(runtime.runtime_archive_path),
        "runtime_archive_sha256": runtime.archive_sha256,
        "runtime_inventory_sha256": runtime.inventory_sha256,
        "runtime_source_bundle_sha256": runtime.source_bundle_sha256,
        "runtime_python_sha256": runtime.python_sha256,
        "bwrap_path": str(runtime.bwrap_path),
        "bwrap_sha256": runtime.bwrap_sha256,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "claim_runtime_verification_receipt_path": str(
            claim_receipt_path.resolve()
        ),
        "runtime_receipt_paths": {
            stage: str(runtime_receipt_outputs[stage].resolve())
            for stage in STAGES
        },
        "input_paths": {
            "checkpoint": str(checkpoint_path.resolve()),
            "command_tokens": str(command_path.resolve()),
            "compiler_weights": str(compiler_path.resolve()),
            "configuration": str(config_path.resolve()),
            "query_reader_weights": str(reader_path.resolve()),
            "query_tokens": str(query_path.resolve()),
            "reactor_weights": str(reactor_path.resolve()),
            "world_tokens": str(world_path.resolve()),
        },
        "custody_paths": {
            "model_assembly_receipt": str(assembly_path.resolve()),
            "tokenization_receipt": str(tokenization_path.resolve()),
            "tokenizer": str(tokenizer_path.resolve()),
        },
        "run_root": str((output_root / "run").resolve()),
        "total_parameters": assembly.total_parameters,
        "architecture_parameters": assembly.architecture_parameters,
        "training_assets_read": False,
    }
    plan_path = output_root / "fixture-plan.json"
    plan_sha256 = _write_json_once(plan_path, plan)
    report = {
        "schema": REPORT_SCHEMA,
        "phase": "prepared",
        "fixture_schema": FIXTURE_SCHEMA,
        "plan_sha256": plan_sha256,
        "manifest_sha256": manifest_sha256,
        "board_sha256": board.receipt.payload_sha256,
        "model_sha256": assembly.complete_model_sha256,
        "checkpoint_sha256": assembly.checkpoint_sha256,
        "checkpoint_step": CHECKPOINT_STEP,
        "total_parameters": assembly.total_parameters,
        "architecture_parameters": assembly.architecture_parameters,
        "training_assets_read": False,
        "public_admission": "pending",
    }
    _write_json_once(output_root / "preparation-report.json", report)
    return {
        **report,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
    }


_PLAN_KEYS = {
    "schema",
    "fixture_schema",
    "checkpoint_kind",
    "checkpoint_step",
    "model_seed",
    "runtime_archive_path",
    "runtime_archive_sha256",
    "runtime_inventory_sha256",
    "runtime_source_bundle_sha256",
    "runtime_python_sha256",
    "bwrap_path",
    "bwrap_sha256",
    "manifest_path",
    "manifest_sha256",
    "claim_runtime_verification_receipt_path",
    "runtime_receipt_paths",
    "input_paths",
    "custody_paths",
    "run_root",
    "total_parameters",
    "architecture_parameters",
    "training_assets_read",
}
_INPUT_KEYS = {
    "checkpoint",
    "command_tokens",
    "compiler_weights",
    "configuration",
    "query_reader_weights",
    "query_tokens",
    "reactor_weights",
    "world_tokens",
}
_CUSTODY_KEYS = {
    "model_assembly_receipt",
    "tokenization_receipt",
    "tokenizer",
}


def validate_plan(plan: Mapping[str, object]) -> None:
    """Reject any widening of the synthetic execution plan."""

    input_paths = plan.get("input_paths")
    custody_paths = plan.get("custody_paths")
    runtime_receipts = plan.get("runtime_receipt_paths")
    if (
        set(plan) != _PLAN_KEYS
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("fixture_schema") != FIXTURE_SCHEMA
        or plan.get("checkpoint_kind")
        != "deterministic-synthetic-untrained"
        or plan.get("checkpoint_step") != CHECKPOINT_STEP
        or plan.get("model_seed") != MODEL_SEED
        or plan.get("training_assets_read") is not False
        or not isinstance(input_paths, dict)
        or set(input_paths) != _INPUT_KEYS
        or not isinstance(custody_paths, dict)
        or set(custody_paths) != _CUSTODY_KEYS
        or not isinstance(runtime_receipts, dict)
        or set(runtime_receipts) != set(STAGES)
        or any(
            not isinstance(plan.get(name), str)
            or _SHA256.fullmatch(str(plan[name])) is None
            for name in (
                "runtime_archive_sha256",
                "runtime_inventory_sha256",
                "runtime_source_bundle_sha256",
                "runtime_python_sha256",
                "bwrap_sha256",
                "manifest_sha256",
            )
        )
        or any(
            not isinstance(value, str) or not Path(value).is_absolute()
            for value in (
                plan["runtime_archive_path"],
                plan["bwrap_path"],
                plan["manifest_path"],
                plan["claim_runtime_verification_receipt_path"],
                plan["run_root"],
                *input_paths.values(),
                *custody_paths.values(),
                *runtime_receipts.values(),
            )
        )
    ):
        raise ETTRSupervisorSmokeError("smoke plan differs")


def _stage_paths(
    plan: Mapping[str, object],
    *,
    stage: str,
    run_root: Path,
) -> tuple[Path, Path, dict[str, Path]]:
    inputs = {
        role: Path(value)
        for role, value in dict(plan["input_paths"]).items()
    }
    manifest_path = Path(str(plan["manifest_path"]))
    if stage == "world":
        direct = {
            "checkpoint": inputs["checkpoint"],
            "compiler_weights": inputs["compiler_weights"],
            "configuration": inputs["configuration"],
            "execution_manifest": manifest_path,
            "world_tokens": inputs["world_tokens"],
        }
    elif stage == "command":
        direct = {
            "checkpoint": inputs["checkpoint"],
            "command_tokens": inputs["command_tokens"],
            "compiled_state": run_root / "world/compiled-state.safetensors",
            "compiler_receipt": run_root / "world/compiler-receipt.json",
            "configuration": inputs["configuration"],
            "execution_manifest": manifest_path,
            "reactor_weights": inputs["reactor_weights"],
        }
    elif stage == "query":
        direct = {
            "checkpoint": inputs["checkpoint"],
            "configuration": inputs["configuration"],
            "execution_manifest": manifest_path,
            "executor_receipt": run_root / "command/executor-receipt.json",
            "query_reader_weights": inputs["query_reader_weights"],
            "query_tokens": inputs["query_tokens"],
            "terminal_state": run_root / "command/terminal-state.safetensors",
        }
    else:
        raise ETTRSupervisorSmokeError("stage differs")
    return (
        run_root / stage,
        run_root / f"launch-{stage}.json",
        direct,
    )


def supervisor_command(
    *,
    host_python: Path,
    runtime_root: str,
    plan: Mapping[str, object],
    stage: str,
    run_root: Path,
    key_descriptor: int,
    allocated_gpu_minor: int,
    run_id: str,
    parent_receipt_sha256: str | None,
) -> list[str]:
    """Construct only the production supervisor's narrow CLI."""

    output_root, receipt_path, direct = _stage_paths(
        plan,
        stage=stage,
        run_root=run_root,
    )
    command = [
        str(host_python),
        "-I",
        "-S",
        "-B",
        "-c",
        _SUPERVISOR_LOADER_CODE,
        runtime_root,
        "--runtime-root",
        runtime_root,
        "--stage",
        stage,
        "--manifest",
        str(plan["manifest_path"]),
        "--manifest-sha256",
        str(plan["manifest_sha256"]),
        "--runtime-archive",
        str(plan["runtime_archive_path"]),
        "--runtime-receipt",
        str(dict(plan["runtime_receipt_paths"])[stage]),
        "--output-root",
        str(output_root),
        "--launch-receipt-output",
        str(receipt_path),
        "--bwrap",
        str(plan["bwrap_path"]),
        "--allocated-gpu-minor",
        str(allocated_gpu_minor),
        "--timeout-seconds",
        "1800",
        "--verifier-private-key-fd",
        str(key_descriptor),
        "--run-id",
        run_id,
    ]
    if parent_receipt_sha256 is not None:
        command.extend(
            (
                "--parent-launch-receipt-sha256",
                parent_receipt_sha256,
            )
        )
    for role, path in direct.items():
        command.extend(("--input", f"{role}={path}"))
    return command


def _sealed_memfd(name: str, payload: bytes) -> int:
    if (
        not hasattr(os, "memfd_create")
        or not hasattr(fcntl, "F_ADD_SEALS")
        or len(payload) != 32
    ):
        raise ETTRSupervisorSmokeError(
            "sealed verifier memfd is unavailable"
        )
    descriptor = os.memfd_create(
        name,
        flags=os.MFD_ALLOW_SEALING,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        fcntl.fcntl(
            descriptor,
            fcntl.F_ADD_SEALS,
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE,
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _load_verified_extractor(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    payload = _read_pinned_source(
        path,
        expected_sha256=expected_sha256,
        max_bytes=4 * 1024 * 1024,
    )
    source_descriptor = os.memfd_create(
        "ettr-extractor-source-bytes",
        flags=os.MFD_ALLOW_SEALING,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(source_descriptor, payload[offset:])
            if written <= 0:
                raise ETTRSupervisorSmokeError(
                    "extractor source write stalled"
                )
            offset += written
        fcntl.fcntl(
            source_descriptor,
            fcntl.F_ADD_SEALS,
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE,
        )
        namespace = runpy.run_path(
            f"/proc/self/fd/{source_descriptor}",
            run_name="_ettr_claim_runtime_smoke",
        )
    finally:
        os.close(source_descriptor)
    if (
        "extract_verified_archive" not in namespace
        or namespace.get("SUPERVISOR_LOADER_CODE")
        != _SUPERVISOR_LOADER_CODE
    ):
        raise ETTRSupervisorSmokeError("trusted extractor API differs")
    return namespace


def _validate_root_owned_python(
    path: Path,
    *,
    expected_sha256: str,
) -> None:
    metadata = path.lstat()
    digest, _ = _sha256_file(path)
    if (
        metadata.st_uid != 0
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o022
        or not metadata.st_mode & 0o111
        or digest != expected_sha256
    ):
        raise ETTRSupervisorSmokeError("trusted host Python differs")


def run_chain(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    trusted_extractor_path: Path,
    expected_trusted_extractor_sha256: str,
    trusted_host_python: Path,
    expected_trusted_host_python_sha256: str,
    extraction_destination: Path,
    allocated_gpu_minor: int,
) -> dict[str, object]:
    """Execute all three stages under one verified extraction and key."""

    if (
        _SHA256.fullmatch(expected_plan_sha256) is None
        or _SHA256.fullmatch(expected_trusted_extractor_sha256) is None
        or _SHA256.fullmatch(expected_trusted_host_python_sha256) is None
        or isinstance(allocated_gpu_minor, bool)
        or not isinstance(allocated_gpu_minor, int)
        or allocated_gpu_minor < 0
        or extraction_destination.exists()
        or extraction_destination.is_symlink()
    ):
        raise ETTRSupervisorSmokeError("chain launch arguments differ")
    plan = _read_canonical_json(
        plan_path,
        expected_sha256=expected_plan_sha256,
        max_bytes=256 * 1024,
    )
    validate_plan(plan)
    _validate_root_owned_python(
        trusted_host_python,
        expected_sha256=expected_trusted_host_python_sha256,
    )
    run_root = Path(str(plan["run_root"]))
    if run_root.exists() or run_root.is_symlink():
        raise ETTRSupervisorSmokeError("run root already exists")
    run_root.mkdir(mode=0o700)
    extractor = _load_verified_extractor(
        trusted_extractor_path,
        expected_sha256=expected_trusted_extractor_sha256,
    )
    extract_verified_archive = extractor["extract_verified_archive"]
    if not callable(extract_verified_archive):
        raise ETTRSupervisorSmokeError("trusted extraction callable differs")

    private_key_descriptor = _sealed_memfd(
        "ettr-launch-verifier-key",
        os.urandom(32),
    )
    run_id = _sha256_bytes(
        os.urandom(32) + bytes.fromhex(str(plan["manifest_sha256"]))
    )
    receipt_hashes: dict[str, str] = {}
    launch_public_key_hex = ""
    try:
        def verified_callback(
            destination_descriptor: int,
            _: object,
        ) -> None:
            nonlocal launch_public_key_hex
            runtime_root = (
                f"/proc/self/fd/{destination_descriptor}/runtime"
            )
            public = subprocess.run(
                (
                    str(trusted_host_python),
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    _PUBLIC_KEY_LOADER_CODE,
                    runtime_root,
                    str(private_key_descriptor),
                ),
                check=True,
                capture_output=True,
                env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"},
                pass_fds=(
                    destination_descriptor,
                    private_key_descriptor,
                ),
                text=True,
            ).stdout.strip()
            if _HEX_32_BYTES.fullmatch(public) is None:
                raise ETTRSupervisorSmokeError(
                    "launch verifier public key differs"
                )
            launch_public_key_hex = public
            parent: str | None = None
            for stage in STAGES:
                command = supervisor_command(
                    host_python=trusted_host_python,
                    runtime_root=runtime_root,
                    plan=plan,
                    stage=stage,
                    run_root=run_root,
                    key_descriptor=private_key_descriptor,
                    allocated_gpu_minor=allocated_gpu_minor,
                    run_id=run_id,
                    parent_receipt_sha256=parent,
                )
                subprocess.run(
                    command,
                    check=True,
                    env={
                        "HOME": "/nonexistent",
                        "PATH": "/usr/bin:/bin",
                    },
                    pass_fds=(
                        destination_descriptor,
                        private_key_descriptor,
                    ),
                )
                receipt_path = run_root / f"launch-{stage}.json"
                receipt_payload = _read_immutable(
                    receipt_path,
                    max_bytes=1024 * 1024,
                )
                try:
                    receipt = json.loads(receipt_payload.decode("ascii"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ETTRSupervisorSmokeError(
                        "launch receipt is malformed"
                    ) from exc
                if (
                    not isinstance(receipt, dict)
                    or receipt_payload != _canonical_json_bytes(receipt)
                    or receipt.get("stage") != stage
                    or receipt.get("run_id") != run_id
                    or receipt.get("parent_launch_receipt_sha256")
                    != parent
                ):
                    raise ETTRSupervisorSmokeError(
                        "launch receipt lineage differs"
                    )
                parent = _sha256_bytes(receipt_payload)
                receipt_hashes[stage] = parent

        extract_verified_archive(
            Path(str(plan["runtime_archive_path"])),
            extraction_destination,
            expected_archive_sha256=str(
                plan["runtime_archive_sha256"]
            ),
            expected_inventory_sha256=str(
                plan["runtime_inventory_sha256"]
            ),
            expected_source_bundle_sha256=str(
                plan["runtime_source_bundle_sha256"]
            ),
            verified_tree_callback=verified_callback,
            remove_after_callback=True,
        )
    finally:
        os.close(private_key_descriptor)
    if (
        tuple(receipt_hashes) != STAGES
        or extraction_destination.exists()
        or _HEX_32_BYTES.fullmatch(launch_public_key_hex) is None
    ):
        raise ETTRSupervisorSmokeError("verified chain completion differs")
    chain = {
        "schema": CHAIN_SCHEMA,
        "run_id": run_id,
        "manifest_sha256": plan["manifest_sha256"],
        "launch_verifier_public_key_hex": launch_public_key_hex,
        "launch_verifier_public_key_sha256": _sha256_bytes(
            bytes.fromhex(launch_public_key_hex)
        ),
        "launch_receipt_sha256s": receipt_hashes,
        "stages": list(STAGES),
        "sealed_memfd_key": True,
        "single_verified_extraction": True,
        "verified_runtime_removed": True,
    }
    chain_path = run_root / "chain.json"
    chain_sha256 = _write_json_once(chain_path, chain)
    return {
        **chain,
        "chain_path": str(chain_path),
        "chain_sha256": chain_sha256,
    }


def _load_verification_receipt(path: Path) -> object:
    from ettr_claim_runtime import (  # noqa: PLC0415
        ETTRClaimRuntimeVerificationReceipt,
    )

    value = _read_canonical_json(path, max_bytes=64 * 1024)
    try:
        receipt = ETTRClaimRuntimeVerificationReceipt(**value)
    except TypeError as exc:
        raise ETTRSupervisorSmokeError(
            "claim runtime receipt differs"
        ) from exc
    receipt.validate()
    return receipt


def validate_chain_and_public_admission(
    *,
    source_root: Path,
    plan_path: Path,
    expected_plan_sha256: str,
) -> dict[str, object]:
    """Authenticate launch receipts and exercise the public admission path."""

    _install_production_paths(source_root)
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: PLC0415
        Ed25519PrivateKey,
    )
    from ettr_deployment_contract import (  # noqa: PLC0415
        ETTRRuntimeImageIdentity,
        ETTRStageLaunchReceipt,
        ETTRStagePolicySpec,
        validate_stage_launch_receipt_chain,
    )
    from ettr_factorial_authority import (  # noqa: PLC0415
        make_root_signed_ettr_custody_authority,
        write_ettr_custody_authority_once,
    )
    from ettr_factorial_custody import (  # noqa: PLC0415
        ETTRFactorialExecutionManifest,
        ETTRLateQueryExecutionReceipt,
        ETTRStageExecutionReceipt,
    )
    from ettr_factorial_qualification import (  # noqa: PLC0415
        bind_terminal_state_artifact,
        materialize_ettr_factorial_qualification,
        materialize_signed_ettr_factorial_qualification,
    )
    from ettr_factorial_qualification_board import (  # noqa: PLC0415
        build_ettr_factorial_qualification_board,
    )
    from ettr_factorial_signed_custody import (  # noqa: PLC0415
        ETTRSignedQualificationAdmission,
        sign_validated_custody_chain,
    )
    from ettr_factorial_tokenization import (  # noqa: PLC0415
        ETTRFactorialTokenizationReceipt,
    )
    from ettr_model_assembly import ETTRModelAssemblyReceipt  # noqa: PLC0415
    from ettr_state_io import read_state  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    plan = _read_canonical_json(
        plan_path,
        expected_sha256=expected_plan_sha256,
        max_bytes=256 * 1024,
    )
    validate_plan(plan)
    run_root = Path(str(plan["run_root"]))
    chain = _read_canonical_json(
        run_root / "chain.json",
        max_bytes=256 * 1024,
    )
    if (
        set(chain)
        != {
            "schema",
            "run_id",
            "manifest_sha256",
            "launch_verifier_public_key_hex",
            "launch_verifier_public_key_sha256",
            "launch_receipt_sha256s",
            "stages",
            "sealed_memfd_key",
            "single_verified_extraction",
            "verified_runtime_removed",
        }
        or chain.get("schema") != CHAIN_SCHEMA
        or chain.get("manifest_sha256") != plan["manifest_sha256"]
        or chain.get("stages") != list(STAGES)
        or chain.get("sealed_memfd_key") is not True
        or chain.get("single_verified_extraction") is not True
        or chain.get("verified_runtime_removed") is not True
    ):
        raise ETTRSupervisorSmokeError("chain report differs")
    launch_public_key = bytes.fromhex(
        str(chain["launch_verifier_public_key_hex"])
    )
    manifest = ETTRFactorialExecutionManifest.from_path(
        Path(str(plan["manifest_path"]))
    )
    manifest.validate_hash(str(plan["manifest_sha256"]))
    claim_receipt = _load_verification_receipt(
        Path(str(plan["claim_runtime_verification_receipt_path"]))
    )
    runtime_identity = ETTRRuntimeImageIdentity.from_manifest(
        asdict(manifest),
        python_sha256=str(plan["runtime_python_sha256"]),
    )
    runtime_identity.validate()
    launches = {
        stage: ETTRStageLaunchReceipt.from_canonical_bytes(
            _read_immutable(
                run_root / f"launch-{stage}.json",
                max_bytes=1024 * 1024,
            ),
            verifier_public_key=launch_public_key,
            runtime_identity=runtime_identity,
            policy=ETTRStagePolicySpec.canonical(stage),
            expected_execution_manifest_sha256=manifest.sha256(),
        )
        for stage in STAGES
    }
    launch_hashes = validate_stage_launch_receipt_chain(
        receipts=launches,
        runtime_identity=runtime_identity,
        expected_execution_manifest_sha256=manifest.sha256(),
        expected_verifier_public_key=launch_public_key,
    )
    expected_launch_hashes = tuple(
        str(dict(chain["launch_receipt_sha256s"])[stage])
        for stage in STAGES
    )
    if launch_hashes != expected_launch_hashes:
        raise ETTRSupervisorSmokeError("launch hash chain differs")

    inputs = {
        role: Path(value)
        for role, value in dict(plan["input_paths"]).items()
    }
    custody_paths = {
        role: Path(value)
        for role, value in dict(plan["custody_paths"]).items()
    }
    compiler_receipt = ETTRStageExecutionReceipt.from_path(
        run_root / "world/compiler-receipt.json"
    )
    executor_receipt = ETTRStageExecutionReceipt.from_path(
        run_root / "command/executor-receipt.json"
    )
    query_receipt = ETTRLateQueryExecutionReceipt(
        **_read_canonical_json(
            run_root / "query/query-receipt.json",
            max_bytes=1024 * 1024,
        )
    )
    tokenization = ETTRFactorialTokenizationReceipt.from_path(
        custody_paths["tokenization_receipt"]
    )
    assembly = ETTRModelAssemblyReceipt.from_path(
        custody_paths["model_assembly_receipt"]
    )
    board = build_ettr_factorial_qualification_board()
    config_payload = _read_canonical_json(
        inputs["configuration"],
        max_bytes=64 * 1024,
    )
    from endogenous_typed_theory_reactor import (  # noqa: PLC0415
        TheoryReactorConfig,
    )

    config = TheoryReactorConfig(**config_payload)
    terminal_path = run_root / "command/terminal-state.safetensors"
    terminal_state = read_state(terminal_path, config)
    artifact = bind_terminal_state_artifact(
        board,
        terminal_state,
        execution_manifest=manifest,
        compiler_receipt=compiler_receipt,
        executor_receipt=executor_receipt,
        expected_model_sha256=assembly.complete_model_sha256,
        expected_execution_manifest_sha256=manifest.sha256(),
        expected_compiler_receipt_sha256=compiler_receipt.sha256(),
        expected_executor_receipt_sha256=executor_receipt.sha256(),
        config=config,
    )
    tokenizer = Tokenizer.from_file(str(custody_paths["tokenizer"]))
    false_token_id = tokenizer.token_to_id("0")
    true_token_id = tokenizer.token_to_id("1")
    if false_token_id is None or true_token_id is None:
        raise ETTRSupervisorSmokeError("synthetic answer codebook differs")
    qualification_batch = materialize_ettr_factorial_qualification(
        board,
        artifact,
        config=config,
        tokenizer=tokenizer,
        tokenizer_sha256=tokenization.tokenizer_sha256,
        vocab_size=256,
        false_token_id=false_token_id,
        true_token_id=true_token_id,
        pad_token_id=tokenization.pad_token_id,
        expected_model_sha256=assembly.complete_model_sha256,
        expected_execution_manifest_sha256=manifest.sha256(),
        expected_compiler_receipt_sha256=compiler_receipt.sha256(),
        expected_executor_receipt_sha256=executor_receipt.sha256(),
    )

    root_key = Ed25519PrivateKey.generate()
    custody_key = Ed25519PrivateKey.generate()
    root_public = root_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    custody_public = custody_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    admission_root = run_root / "public-admission"
    admission_root.mkdir(mode=0o700)
    root_public_path = admission_root / "root-public-key.bin"
    _write_once(root_public_path, root_public)
    authority = make_root_signed_ettr_custody_authority(
        root_private_key=root_key,
        custody_public_key_hex=custody_public.hex(),
        launch_verifier_public_key_hex=launch_public_key.hex(),
        claim_runtime_verification_receipt_sha256=claim_receipt.sha256(),
        board_sha256=board.receipt.payload_sha256,
        execution_manifest_sha256=manifest.sha256(),
    )
    authority_path = admission_root / "authority.json"
    write_ettr_custody_authority_once(authority_path, authority)
    seal = sign_validated_custody_chain(
        board,
        private_key=custody_key,
        authority_record=authority,
        execution_manifest=manifest,
        expected_execution_manifest_sha256=manifest.sha256(),
        tokenization_receipt=tokenization,
        tokenizer_path=custody_paths["tokenizer"],
        model_assembly_receipt=assembly,
        config=config,
        config_path=inputs["configuration"],
        checkpoint_path=inputs["checkpoint"],
        compiler_path=inputs["compiler_weights"],
        reactor_path=inputs["reactor_weights"],
        query_reader_path=inputs["query_reader_weights"],
        compiler_receipt=compiler_receipt,
        expected_compiler_receipt_sha256=compiler_receipt.sha256(),
        executor_receipt=executor_receipt,
        expected_executor_receipt_sha256=executor_receipt.sha256(),
        query_receipt=query_receipt,
        expected_query_receipt_sha256=query_receipt.sha256(),
        claim_runtime_verification_receipt=claim_receipt,
        runtime_identity=runtime_identity,
        world_launch_receipt=launches["world"],
        expected_world_launch_receipt_sha256=launch_hashes[0],
        command_launch_receipt=launches["command"],
        expected_command_launch_receipt_sha256=launch_hashes[1],
        query_launch_receipt=launches["query"],
        expected_query_launch_receipt_sha256=launch_hashes[2],
        terminal_state_path=terminal_path,
        query_tokens_path=inputs["query_tokens"],
        answer_path=run_root / "query/answer.json",
        qualification_batch=qualification_batch,
        qualification_vocab_size=256,
        false_token_id=false_token_id,
        true_token_id=true_token_id,
        pad_token_id=tokenization.pad_token_id,
    )
    signed_admission = ETTRSignedQualificationAdmission(
        query_receipt=query_receipt,
        world_launch_receipt=launches["world"],
        command_launch_receipt=launches["command"],
        query_launch_receipt=launches["query"],
        custody_seal=seal,
    )
    admitted_batch = materialize_signed_ettr_factorial_qualification(
        board,
        artifact,
        signed_admission,
        config=config,
        tokenizer=tokenizer,
        tokenizer_sha256=tokenization.tokenizer_sha256,
        vocab_size=256,
        false_token_id=false_token_id,
        true_token_id=true_token_id,
        pad_token_id=tokenization.pad_token_id,
        expected_model_sha256=assembly.complete_model_sha256,
        expected_execution_manifest_sha256=manifest.sha256(),
        expected_compiler_receipt_sha256=compiler_receipt.sha256(),
        expected_executor_receipt_sha256=executor_receipt.sha256(),
        tokenization_receipt=tokenization,
        tokenizer_path=custody_paths["tokenizer"],
        expected_tokenization_receipt_sha256=tokenization.sha256(),
        expected_query_receipt_sha256=query_receipt.sha256(),
        claim_runtime_verification_receipt=claim_receipt,
        runtime_identity=runtime_identity,
        expected_world_launch_receipt_sha256=launch_hashes[0],
        expected_command_launch_receipt_sha256=launch_hashes[1],
        expected_query_launch_receipt_sha256=launch_hashes[2],
        expected_custody_seal_sha256=seal.sha256(),
        authority_record_path=authority_path,
        root_public_key_path=root_public_path,
        pinned_root_public_key_sha256=_sha256_bytes(root_public),
        expected_authority_record_sha256=authority.sha256(),
    )
    if admitted_batch.sha256() != qualification_batch.sha256():
        raise ETTRSupervisorSmokeError("public admission batch differs")
    _write_json_once(admission_root / "custody-seal.json", asdict(seal))
    report = {
        "schema": REPORT_SCHEMA,
        "phase": "complete",
        "fixture_schema": FIXTURE_SCHEMA,
        "manifest_sha256": manifest.sha256(),
        "run_id": chain["run_id"],
        "world_launch_receipt_sha256": launch_hashes[0],
        "command_launch_receipt_sha256": launch_hashes[1],
        "query_launch_receipt_sha256": launch_hashes[2],
        "compiler_receipt_sha256": compiler_receipt.sha256(),
        "executor_receipt_sha256": executor_receipt.sha256(),
        "query_receipt_sha256": query_receipt.sha256(),
        "custody_seal_sha256": seal.sha256(),
        "qualification_batch_sha256": admitted_batch.sha256(),
        "row_count": manifest.row_count,
        "checkpoint_step": manifest.checkpoint_step,
        "total_parameters": assembly.total_parameters,
        "architecture_parameters": assembly.architecture_parameters,
        "sealed_memfd_key": True,
        "single_verified_extraction": True,
        "network_isolated_stages": 3,
        "signed_launch_chain": "pass",
        "public_admission": "pass",
        "training_assets_read": False,
    }
    report_path = run_root / "smoke-report.json"
    report_sha256 = _write_json_once(report_path, report)
    return {
        **report,
        "report_path": str(report_path),
        "report_sha256": report_sha256,
    }


def _prepare_parser(subparsers: object) -> None:
    parser = subparsers.add_parser("prepare")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--runtime-inventory", type=Path, required=True)
    parser.add_argument(
        "--claim-runtime-verification-receipt",
        type=Path,
        required=True,
    )
    for stage in STAGES:
        parser.add_argument(
            f"--runtime-receipt-{stage}",
            type=Path,
            required=True,
        )
    parser.add_argument("--bwrap", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--expected-source-bundle-sha256", required=True)
    parser.add_argument("--expected-bwrap-sha256", required=True)


def _run_chain_parser(subparsers: object) -> None:
    parser = subparsers.add_parser("run-chain")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--trusted-extractor", type=Path, required=True)
    parser.add_argument("--expected-trusted-extractor-sha256", required=True)
    parser.add_argument("--trusted-host-python", type=Path, required=True)
    parser.add_argument(
        "--expected-trusted-host-python-sha256",
        required=True,
    )
    parser.add_argument("--extraction-destination", type=Path, required=True)
    parser.add_argument("--allocated-gpu-minor", type=int, required=True)


def _validate_parser(subparsers: object) -> None:
    parser = subparsers.add_parser("validate")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _prepare_parser(subparsers)
    _run_chain_parser(subparsers)
    _validate_parser(subparsers)
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare":
        runtime = load_runtime_bindings(
            source_root=arguments.source_root,
            runtime_archive_path=arguments.runtime_archive,
            runtime_inventory_path=arguments.runtime_inventory,
            claim_runtime_verification_receipt_path=(
                arguments.claim_runtime_verification_receipt
            ),
            runtime_receipt_paths={
                stage: getattr(arguments, f"runtime_receipt_{stage}")
                for stage in STAGES
            },
            bwrap_path=arguments.bwrap,
            expected_archive_sha256=arguments.expected_archive_sha256,
            expected_inventory_sha256=arguments.expected_inventory_sha256,
            expected_source_bundle_sha256=(
                arguments.expected_source_bundle_sha256
            ),
            expected_bwrap_sha256=arguments.expected_bwrap_sha256,
        )
        result = prepare_fixture(
            source_root=arguments.source_root,
            output_root=arguments.output_root,
            runtime=runtime,
        )
    elif arguments.command == "run-chain":
        result = run_chain(
            plan_path=arguments.plan,
            expected_plan_sha256=arguments.expected_plan_sha256,
            trusted_extractor_path=arguments.trusted_extractor,
            expected_trusted_extractor_sha256=(
                arguments.expected_trusted_extractor_sha256
            ),
            trusted_host_python=arguments.trusted_host_python,
            expected_trusted_host_python_sha256=(
                arguments.expected_trusted_host_python_sha256
            ),
            extraction_destination=arguments.extraction_destination,
            allocated_gpu_minor=arguments.allocated_gpu_minor,
        )
    else:
        result = validate_chain_and_public_admission(
            source_root=arguments.source_root,
            plan_path=arguments.plan,
            expected_plan_sha256=arguments.expected_plan_sha256,
        )
    print(_canonical_json_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CHAIN_SCHEMA",
    "CHECKPOINT_STEP",
    "ETTRSupervisorSmokeError",
    "FIXTURE_SCHEMA",
    "MODEL_SEED",
    "PLAN_SCHEMA",
    "REPORT_SCHEMA",
    "STAGES",
    "SmokeRuntimeBindings",
    "load_runtime_bindings",
    "prepare_fixture",
    "run_chain",
    "supervisor_command",
    "validate_chain_and_public_admission",
    "validate_plan",
]
