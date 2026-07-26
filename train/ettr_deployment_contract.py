"""Canonical runtime and stage-policy identities for ETTR deployment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Literal, Mapping, Sequence


RUNTIME_IDENTITY_SCHEMA = "ettr-runtime-image-identity-v1"
STAGE_POLICY_SCHEMA = "ettr-stage-policy-v1"
STAGE_LAUNCH_RECEIPT_SCHEMA = "ettr-stage-launch-receipt-v1"
Stage = Literal["world", "command", "query"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NETNS = re.compile(r"^[0-9]+:[0-9]+$")

_READ_ROLES: dict[Stage, tuple[str, ...]] = {
    "world": (
        "application_bundle",
        "checkpoint",
        "compiler_weights",
        "configuration",
        "execution_manifest",
        "runtime_bundle_receipt",
        "runtime_image",
        "world_tokens",
    ),
    "command": (
        "application_bundle",
        "checkpoint",
        "command_tokens",
        "compiled_state",
        "compiler_receipt",
        "configuration",
        "execution_manifest",
        "reactor_weights",
        "runtime_bundle_receipt",
        "runtime_image",
    ),
    "query": (
        "application_bundle",
        "checkpoint",
        "configuration",
        "execution_manifest",
        "executor_receipt",
        "query_reader_weights",
        "query_tokens",
        "runtime_bundle_receipt",
        "runtime_image",
        "terminal_state",
    ),
}
_WRITE_ROLES: dict[Stage, tuple[str, ...]] = {
    "world": (
        "compiled_state_output",
        "compiler_receipt_output",
        "sandbox_receipt_output",
    ),
    "command": (
        "executor_receipt_output",
        "sandbox_receipt_output",
        "terminal_state_output",
    ),
    "query": (
        "answer_output",
        "query_receipt_output",
        "sandbox_receipt_output",
    ),
}


class ETTRDeploymentContractError(ValueError):
    """The measured runtime or stage policy differs from its frozen contract."""


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


def _require_hash(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ETTRDeploymentContractError(f"{label} is not a SHA-256")
    return value


def _validate_role_hashes(
    rows: tuple[tuple[str, str], ...],
    *,
    expected_roles: tuple[str, ...],
    label: str,
) -> None:
    if tuple(role for role, _ in rows) != expected_roles:
        raise ETTRDeploymentContractError(f"{label} role inventory differs")
    for role, digest in rows:
        if not role or _require_hash(digest, f"{label} {role}") != digest:
            raise ETTRDeploymentContractError(f"{label} role identity differs")


@dataclass(frozen=True, slots=True)
class ETTRRuntimeImageIdentity:
    schema: str
    archive_sha256: str
    archive_size: int
    inventory_sha256: str
    world_runtime_bundle_sha256: str
    command_runtime_bundle_sha256: str
    query_runtime_bundle_sha256: str
    python_sha256: str
    bootstrap_sha256: str
    external_launcher_sha256: str
    bwrap_sha256: str
    network_namespace_required: bool

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(asdict(self))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validate(self) -> None:
        if (
            self.schema != RUNTIME_IDENTITY_SCHEMA
            or isinstance(self.archive_size, bool)
            or not isinstance(self.archive_size, int)
            or self.archive_size <= 0
            or not isinstance(self.network_namespace_required, bool)
            or not self.network_namespace_required
        ):
            raise ETTRDeploymentContractError(
                "runtime image identity geometry differs"
            )
        for label, digest in (
            ("archive", self.archive_sha256),
            ("inventory", self.inventory_sha256),
            ("WORLD runtime bundle", self.world_runtime_bundle_sha256),
            ("COMMAND runtime bundle", self.command_runtime_bundle_sha256),
            ("QUERY runtime bundle", self.query_runtime_bundle_sha256),
            ("python", self.python_sha256),
            ("bootstrap", self.bootstrap_sha256),
            ("external launcher", self.external_launcher_sha256),
            ("bubblewrap", self.bwrap_sha256),
        ):
            _require_hash(digest, label)

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, object],
        *,
        python_sha256: str,
    ) -> ETTRRuntimeImageIdentity:
        try:
            identity = cls(
                schema=RUNTIME_IDENTITY_SCHEMA,
                archive_sha256=manifest["claim_runtime_archive_sha256"],
                archive_size=manifest["claim_runtime_archive_size"],
                inventory_sha256=manifest[
                    "claim_runtime_inventory_sha256"
                ],
                world_runtime_bundle_sha256=manifest[
                    "world_runtime_bundle_sha256"
                ],
                command_runtime_bundle_sha256=manifest[
                    "command_runtime_bundle_sha256"
                ],
                query_runtime_bundle_sha256=manifest[
                    "query_runtime_bundle_sha256"
                ],
                python_sha256=python_sha256,
                bootstrap_sha256=manifest["bootstrap_sha256"],
                external_launcher_sha256=manifest[
                    "external_launcher_sha256"
                ],
                bwrap_sha256=manifest["bwrap_sha256"],
                network_namespace_required=manifest[
                    "network_namespace_required"
                ],
            )
        except KeyError as exc:
            raise ETTRDeploymentContractError(
                "runtime image manifest binding is incomplete"
            ) from exc
        identity.validate()
        return identity


@dataclass(frozen=True, slots=True)
class ETTRStagePolicySpec:
    schema: str
    stage: Stage
    read_roles: tuple[str, ...]
    write_roles: tuple[str, ...]
    deny_network: bool
    unshare_pid: bool
    unshare_ipc: bool
    unshare_uts: bool
    fresh_proc: bool
    clean_environment: bool
    close_inherited_fds: bool
    forbid_repository_root: bool
    forbid_assessor: bool
    forbid_signing_keys: bool

    @classmethod
    def canonical(cls, stage: Stage) -> ETTRStagePolicySpec:
        if stage not in _READ_ROLES:
            raise ETTRDeploymentContractError("stage identity differs")
        return cls(
            schema=STAGE_POLICY_SCHEMA,
            stage=stage,
            read_roles=_READ_ROLES[stage],
            write_roles=_WRITE_ROLES[stage],
            deny_network=True,
            unshare_pid=True,
            unshare_ipc=True,
            unshare_uts=True,
            fresh_proc=True,
            clean_environment=True,
            close_inherited_fds=True,
            forbid_repository_root=True,
            forbid_assessor=True,
            forbid_signing_keys=True,
        )

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(asdict(self))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validate(self) -> None:
        expected = self.canonical(self.stage)
        if self != expected:
            raise ETTRDeploymentContractError("stage policy differs")


@dataclass(frozen=True, slots=True)
class ETTRStageLaunchReceipt:
    schema: str
    stage: Stage
    execution_manifest_sha256: str
    runtime_identity_sha256: str
    stage_policy_sha256: str
    bwrap_sha256: str
    input_role_sha256s: tuple[tuple[str, str], ...]
    output_role_sha256s: tuple[tuple[str, str], ...]
    parent_network_namespace: str
    child_network_namespace: str
    environment_sha256: str
    loaded_object_map_sha256: str

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(asdict(self))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validate(
        self,
        *,
        runtime_identity: ETTRRuntimeImageIdentity,
        policy: ETTRStagePolicySpec,
        expected_execution_manifest_sha256: str,
    ) -> None:
        runtime_identity.validate()
        policy.validate()
        if (
            self.schema != STAGE_LAUNCH_RECEIPT_SCHEMA
            or self.stage != policy.stage
            or self.execution_manifest_sha256
            != expected_execution_manifest_sha256
            or self.runtime_identity_sha256 != runtime_identity.sha256()
            or self.stage_policy_sha256 != policy.sha256()
            or self.bwrap_sha256 != runtime_identity.bwrap_sha256
            or _NETNS.fullmatch(self.parent_network_namespace) is None
            or _NETNS.fullmatch(self.child_network_namespace) is None
            or self.parent_network_namespace
            == self.child_network_namespace
        ):
            raise ETTRDeploymentContractError("stage launch receipt differs")
        for label, digest in (
            ("execution manifest", self.execution_manifest_sha256),
            ("runtime identity", self.runtime_identity_sha256),
            ("stage policy", self.stage_policy_sha256),
            ("bubblewrap", self.bwrap_sha256),
            ("environment", self.environment_sha256),
            ("loaded object map", self.loaded_object_map_sha256),
        ):
            _require_hash(digest, label)
        _validate_role_hashes(
            self.input_role_sha256s,
            expected_roles=policy.read_roles,
            label="input",
        )
        _validate_role_hashes(
            self.output_role_sha256s,
            expected_roles=policy.write_roles,
            label="output",
        )


def canonical_stage_policy_sha256s() -> dict[Stage, str]:
    return {
        stage: ETTRStagePolicySpec.canonical(stage).sha256()
        for stage in ("world", "command", "query")
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-stage-policy-sha256s",
        action="store_true",
        required=True,
    )
    parser.parse_args(argv)
    print(
        _canonical_json_bytes(canonical_stage_policy_sha256s()).decode(
            "ascii"
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ETTRDeploymentContractError",
    "ETTRRuntimeImageIdentity",
    "ETTRStageLaunchReceipt",
    "ETTRStagePolicySpec",
    "RUNTIME_IDENTITY_SCHEMA",
    "STAGE_LAUNCH_RECEIPT_SCHEMA",
    "STAGE_POLICY_SCHEMA",
    "canonical_stage_policy_sha256s",
]
