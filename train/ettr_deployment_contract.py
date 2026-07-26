"""Canonical runtime and stage-policy identities for ETTR deployment."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import re
from typing import Literal, Mapping, Sequence


RUNTIME_IDENTITY_SCHEMA = "ettr-runtime-image-identity-v1"
STAGE_POLICY_SCHEMA = "ettr-stage-policy-v1"
STAGE_LAUNCH_RECEIPT_SCHEMA = "ettr-stage-launch-receipt-v3"
STAGE_LAUNCH_SIGNATURE_DOMAIN = (
    b"shohin-ettr-stage-launch-receipt-v3\x00"
)
Stage = Literal["world", "command", "query"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_64_BYTES = re.compile(r"^[0-9a-f]{128}$")
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
    ),
    "command": (
        "executor_receipt_output",
        "terminal_state_output",
    ),
    "query": (
        "answer_output",
        "query_receipt_output",
    ),
}
_ROLE_DESTINATIONS: dict[str, str] = {
    "application_bundle": "/app",
    "checkpoint": "/inputs/checkpoint.pt",
    "command_tokens": "/inputs/command.json",
    "compiled_state": "/inputs/compiled-state.safetensors",
    "compiler_receipt": "/inputs/compiler-receipt.json",
    "compiler_weights": "/inputs/compiler.safetensors",
    "configuration": "/inputs/config.json",
    "execution_manifest": "/inputs/execution-manifest.json",
    "executor_receipt": "/inputs/executor-receipt.json",
    "query_reader_weights": "/inputs/query-reader.safetensors",
    "query_tokens": "/inputs/query.json",
    "reactor_weights": "/inputs/reactor.safetensors",
    "runtime_bundle_receipt": "/inputs/runtime-receipt.json",
    "runtime_image": "/runtime",
    "terminal_state": "/inputs/terminal-state.safetensors",
    "world_tokens": "/inputs/world.json",
    "answer_output": "/outputs/answer.json",
    "compiled_state_output": "/outputs/compiled-state.safetensors",
    "compiler_receipt_output": "/outputs/compiler-receipt.json",
    "executor_receipt_output": "/outputs/executor-receipt.json",
    "query_receipt_output": "/outputs/query-receipt.json",
    "terminal_state_output": "/outputs/terminal-state.safetensors",
}


class ETTRDeploymentContractError(ValueError):
    """The measured runtime or stage policy differs from its frozen contract."""


@lru_cache(maxsize=1)
def _openssl_ed25519() -> ctypes.CDLL:
    """Load the verifier host's EVP Ed25519 implementation."""

    library_name = ctypes.util.find_library("crypto")
    if library_name is None:
        raise ETTRDeploymentContractError(
            "verifier Ed25519 implementation is unavailable"
        )
    try:
        library = ctypes.CDLL(library_name)
        library.OBJ_txt2nid.argtypes = [ctypes.c_char_p]
        library.OBJ_txt2nid.restype = ctypes.c_int
        library.EVP_PKEY_new_raw_private_key.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.EVP_PKEY_new_raw_private_key.restype = ctypes.c_void_p
        library.EVP_PKEY_new_raw_public_key.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.EVP_PKEY_new_raw_public_key.restype = ctypes.c_void_p
        library.EVP_PKEY_get_raw_public_key.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        library.EVP_PKEY_get_raw_public_key.restype = ctypes.c_int
        library.EVP_MD_CTX_new.argtypes = []
        library.EVP_MD_CTX_new.restype = ctypes.c_void_p
        library.EVP_DigestSignInit.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library.EVP_DigestSignInit.restype = ctypes.c_int
        library.EVP_DigestSign.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.EVP_DigestSign.restype = ctypes.c_int
        library.EVP_DigestVerifyInit.argtypes = (
            library.EVP_DigestSignInit.argtypes
        )
        library.EVP_DigestVerifyInit.restype = ctypes.c_int
        library.EVP_DigestVerify.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.EVP_DigestVerify.restype = ctypes.c_int
        library.EVP_MD_CTX_free.argtypes = [ctypes.c_void_p]
        library.EVP_MD_CTX_free.restype = None
        library.EVP_PKEY_free.argtypes = [ctypes.c_void_p]
        library.EVP_PKEY_free.restype = None
    except AttributeError as exc:
        raise ETTRDeploymentContractError(
            "verifier Ed25519 implementation differs"
        ) from exc
    if library.OBJ_txt2nid(b"ED25519") <= 0:
        raise ETTRDeploymentContractError(
            "verifier Ed25519 algorithm is unavailable"
        )
    return library


def _ed25519_public_key_ctypes(private_key_bytes: bytes) -> bytes:
    library = _openssl_ed25519()
    private_buffer = ctypes.create_string_buffer(private_key_bytes)
    key = library.EVP_PKEY_new_raw_private_key(
        library.OBJ_txt2nid(b"ED25519"),
        None,
        private_buffer,
        len(private_key_bytes),
    )
    if not key:
        raise ETTRDeploymentContractError("verifier private key differs")
    try:
        public_size = ctypes.c_size_t(32)
        public_buffer = ctypes.create_string_buffer(32)
        if (
            library.EVP_PKEY_get_raw_public_key(
                key,
                public_buffer,
                ctypes.byref(public_size),
            )
            != 1
            or public_size.value != 32
        ):
            raise ETTRDeploymentContractError(
                "verifier public key derivation failed"
            )
        return public_buffer.raw
    finally:
        library.EVP_PKEY_free(key)


def _ed25519_sign_ctypes(private_key_bytes: bytes, payload: bytes) -> bytes:
    library = _openssl_ed25519()
    private_buffer = ctypes.create_string_buffer(private_key_bytes)
    key = library.EVP_PKEY_new_raw_private_key(
        library.OBJ_txt2nid(b"ED25519"),
        None,
        private_buffer,
        len(private_key_bytes),
    )
    if not key:
        raise ETTRDeploymentContractError("verifier private key differs")
    context = library.EVP_MD_CTX_new()
    if not context:
        library.EVP_PKEY_free(key)
        raise ETTRDeploymentContractError(
            "verifier signature context failed"
        )
    try:
        payload_buffer = ctypes.create_string_buffer(payload)
        signature_size = ctypes.c_size_t(64)
        signature_buffer = ctypes.create_string_buffer(64)
        if (
            library.EVP_DigestSignInit(
                context,
                None,
                None,
                None,
                key,
            )
            != 1
            or library.EVP_DigestSign(
                context,
                signature_buffer,
                ctypes.byref(signature_size),
                payload_buffer,
                len(payload),
            )
            != 1
            or signature_size.value != 64
        ):
            raise ETTRDeploymentContractError(
                "verifier signature operation failed"
            )
        return signature_buffer.raw
    finally:
        library.EVP_MD_CTX_free(context)
        library.EVP_PKEY_free(key)


def _ed25519_verify_ctypes(
    public_key_bytes: bytes,
    signature: bytes,
    payload: bytes,
) -> None:
    library = _openssl_ed25519()
    public_buffer = ctypes.create_string_buffer(public_key_bytes)
    key = library.EVP_PKEY_new_raw_public_key(
        library.OBJ_txt2nid(b"ED25519"),
        None,
        public_buffer,
        len(public_key_bytes),
    )
    if not key:
        raise ETTRDeploymentContractError("verifier public key differs")
    context = library.EVP_MD_CTX_new()
    if not context:
        library.EVP_PKEY_free(key)
        raise ETTRDeploymentContractError(
            "verifier signature context failed"
        )
    try:
        signature_buffer = ctypes.create_string_buffer(signature)
        payload_buffer = ctypes.create_string_buffer(payload)
        if (
            library.EVP_DigestVerifyInit(
                context,
                None,
                None,
                None,
                key,
            )
            != 1
            or library.EVP_DigestVerify(
                context,
                signature_buffer,
                len(signature),
                payload_buffer,
                len(payload),
            )
            != 1
        ):
            raise ETTRDeploymentContractError(
                "verifier launch signature differs"
            )
    finally:
        library.EVP_MD_CTX_free(context)
        library.EVP_PKEY_free(key)


def ed25519_public_key_from_private_bytes(
    private_key_bytes: bytes,
) -> bytes:
    """Derive one raw Ed25519 public key without importing candidate code."""

    if not isinstance(private_key_bytes, bytes) or len(private_key_bytes) != 32:
        raise ETTRDeploymentContractError("verifier private key differs")
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError:
        return _ed25519_public_key_ctypes(private_key_bytes)
    return (
        Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )


def ed25519_sign(
    private_key_bytes: bytes,
    payload: bytes,
) -> bytes:
    """Sign verifier-owned bytes with a raw Ed25519 private seed."""

    if (
        not isinstance(private_key_bytes, bytes)
        or len(private_key_bytes) != 32
        or not isinstance(payload, bytes)
    ):
        raise ETTRDeploymentContractError("verifier signature input differs")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError:
        return _ed25519_sign_ctypes(private_key_bytes, payload)
    return Ed25519PrivateKey.from_private_bytes(private_key_bytes).sign(payload)


def ed25519_verify(
    public_key_bytes: bytes,
    signature: bytes,
    payload: bytes,
) -> None:
    """Verify a raw Ed25519 signature with the verifier public key."""

    if (
        not isinstance(public_key_bytes, bytes)
        or len(public_key_bytes) != 32
        or not isinstance(signature, bytes)
        or len(signature) != 64
        or not isinstance(payload, bytes)
    ):
        raise ETTRDeploymentContractError("verifier signature input differs")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError:
        _ed25519_verify_ctypes(public_key_bytes, signature, payload)
        return
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature,
            payload,
        )
    except (InvalidSignature, ValueError) as exc:
        raise ETTRDeploymentContractError(
            "verifier launch signature differs"
        ) from exc


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


def canonical_stage_environment(
    *,
    stage: Stage,
    runtime_identity: ETTRRuntimeImageIdentity,
    allocated_gpu_minor: int,
) -> tuple[tuple[str, str], ...]:
    """Exact environment passed to Bubblewrap and the verified bootstrap."""

    if (
        stage not in _READ_ROLES
        or isinstance(allocated_gpu_minor, bool)
        or not isinstance(allocated_gpu_minor, int)
        or allocated_gpu_minor < 0
    ):
        raise ETTRDeploymentContractError("stage environment input differs")
    policy = ETTRStagePolicySpec.canonical(stage)
    return tuple(
        sorted(
            {
                "CUDA_VISIBLE_DEVICES": str(allocated_gpu_minor),
                "ETTR_ALLOCATED_GPU_INDEX": str(allocated_gpu_minor),
                "HOME": "/tmp",
                "LD_LIBRARY_PATH": (
                    "/runtime/miniforge3/lib:/usr/lib64:/lib64:/usr/lib:/lib"
                ),
                "PATH": "/runtime/miniforge3/bin",
                "PYTHONNOUSERSITE": "1",
                "SHOHIN_ETTR_BWRAP_SHA256": runtime_identity.bwrap_sha256,
                "SHOHIN_ETTR_CLAIM_RUNTIME_INVENTORY_SHA256": (
                    runtime_identity.inventory_sha256
                ),
                "SHOHIN_ETTR_CLAIM_RUNTIME_SHA256": (
                    runtime_identity.archive_sha256
                ),
                "SHOHIN_ETTR_EXTERNAL_LAUNCHER_SHA256": (
                    runtime_identity.external_launcher_sha256
                ),
                "SHOHIN_ETTR_NETWORK_NAMESPACE_ISOLATED": "1",
                "SHOHIN_ETTR_STAGE_POLICY_SHA256": policy.sha256(),
            }.items()
        )
    )


def canonical_stage_environment_sha256(
    *,
    stage: Stage,
    runtime_identity: ETTRRuntimeImageIdentity,
    allocated_gpu_minor: int,
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            canonical_stage_environment(
                stage=stage,
                runtime_identity=runtime_identity,
                allocated_gpu_minor=allocated_gpu_minor,
            )
        )
    ).hexdigest()


def canonical_loaded_object_map_sha256(
    *,
    stage: Stage,
    runtime_identity: ETTRRuntimeImageIdentity,
    input_role_sha256s: tuple[tuple[str, str], ...],
    output_role_sha256s: tuple[tuple[str, str], ...],
) -> str:
    """Hash the exact logical object/mount map admitted for one stage."""

    policy = ETTRStagePolicySpec.canonical(stage)
    _validate_role_hashes(
        input_role_sha256s,
        expected_roles=policy.read_roles,
        label="input",
    )
    _validate_role_hashes(
        output_role_sha256s,
        expected_roles=policy.write_roles,
        label="output",
    )
    rows = [
        {
            "destination": _ROLE_DESTINATIONS[role],
            "mode": "read-only",
            "role": role,
            "sha256": digest,
        }
        for role, digest in input_role_sha256s
    ]
    rows.extend(
        {
            "destination": _ROLE_DESTINATIONS[role],
            "mode": "write-only-result",
            "role": role,
            "sha256": digest,
        }
        for role, digest in output_role_sha256s
    )
    rows.extend(
        (
            {
                "destination": "/runtime/app/tools/run_ettr_verified_stage.py",
                "mode": "verified-bootstrap",
                "role": "bootstrap",
                "sha256": runtime_identity.bootstrap_sha256,
            },
            {
                "destination": "/usr/bin/bwrap",
                "mode": "external-launcher",
                "role": "bubblewrap",
                "sha256": runtime_identity.bwrap_sha256,
            },
            {
                "destination": "<supervisor>",
                "mode": "external-supervisor",
                "role": "supervisor",
                "sha256": runtime_identity.external_launcher_sha256,
            },
        )
    )
    return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


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
    run_id: str
    parent_launch_receipt_sha256: str | None
    verifier_public_key_sha256: str
    execution_manifest_sha256: str
    runtime_identity_sha256: str
    stage_policy_sha256: str
    bwrap_sha256: str
    input_role_sha256s: tuple[tuple[str, str], ...]
    output_role_sha256s: tuple[tuple[str, str], ...]
    parent_network_namespace: str
    child_network_namespace: str
    allocated_gpu_minor: int
    exit_code: int
    stdout_sha256: str
    stderr_sha256: str
    environment_sha256: str
    loaded_object_map_sha256: str
    verifier_signature_hex: str

    def unsigned_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["verifier_signature_hex"] = ""
        return payload

    def signing_bytes(self) -> bytes:
        return STAGE_LAUNCH_SIGNATURE_DOMAIN + _canonical_json_bytes(
            self.unsigned_payload()
        )

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
        expected_verifier_public_key: bytes | None = None,
    ) -> None:
        runtime_identity.validate()
        policy.validate()
        if (
            self.schema != STAGE_LAUNCH_RECEIPT_SCHEMA
            or self.stage != policy.stage
            or _SHA256.fullmatch(self.run_id) is None
            or (
                self.stage == "world"
                and self.parent_launch_receipt_sha256 is not None
            )
            or (
                self.stage != "world"
                and (
                    not isinstance(
                        self.parent_launch_receipt_sha256,
                        str,
                    )
                    or _SHA256.fullmatch(
                        self.parent_launch_receipt_sha256
                    )
                    is None
                )
            )
            or self.execution_manifest_sha256
            != expected_execution_manifest_sha256
            or self.runtime_identity_sha256 != runtime_identity.sha256()
            or self.stage_policy_sha256 != policy.sha256()
            or self.bwrap_sha256 != runtime_identity.bwrap_sha256
            or _NETNS.fullmatch(self.parent_network_namespace) is None
            or _NETNS.fullmatch(self.child_network_namespace) is None
            or self.parent_network_namespace
            == self.child_network_namespace
            or isinstance(self.allocated_gpu_minor, bool)
            or not isinstance(self.allocated_gpu_minor, int)
            or self.allocated_gpu_minor < 0
            or isinstance(self.exit_code, bool)
            or self.exit_code != 0
        ):
            raise ETTRDeploymentContractError("stage launch receipt differs")
        for label, digest in (
            ("execution manifest", self.execution_manifest_sha256),
            ("runtime identity", self.runtime_identity_sha256),
            ("stage policy", self.stage_policy_sha256),
            ("bubblewrap", self.bwrap_sha256),
            ("stdout", self.stdout_sha256),
            ("stderr", self.stderr_sha256),
            ("environment", self.environment_sha256),
            ("loaded object map", self.loaded_object_map_sha256),
            ("verifier public key", self.verifier_public_key_sha256),
        ):
            _require_hash(digest, label)
        if _HEX_64_BYTES.fullmatch(self.verifier_signature_hex) is None:
            raise ETTRDeploymentContractError(
                "verifier launch signature differs"
            )
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
        if (
            self.environment_sha256
            != canonical_stage_environment_sha256(
                stage=self.stage,
                runtime_identity=runtime_identity,
                allocated_gpu_minor=self.allocated_gpu_minor,
            )
            or self.loaded_object_map_sha256
            != canonical_loaded_object_map_sha256(
                stage=self.stage,
                runtime_identity=runtime_identity,
                input_role_sha256s=self.input_role_sha256s,
                output_role_sha256s=self.output_role_sha256s,
            )
        ):
            raise ETTRDeploymentContractError(
                "measured stage launch map differs"
            )
        if expected_verifier_public_key is not None:
            self.verify_verifier_signature(expected_verifier_public_key)

    def verify_verifier_signature(
        self,
        verifier_public_key: bytes,
    ) -> None:
        if (
            not isinstance(verifier_public_key, bytes)
            or len(verifier_public_key) != 32
            or hashlib.sha256(verifier_public_key).hexdigest()
            != self.verifier_public_key_sha256
        ):
            raise ETTRDeploymentContractError(
                "verifier public key identity differs"
            )
        try:
            signature = bytes.fromhex(self.verifier_signature_hex)
        except ValueError as exc:
            raise ETTRDeploymentContractError(
                "verifier launch signature differs"
            ) from exc
        ed25519_verify(
            verifier_public_key,
            signature,
            self.signing_bytes(),
        )

    @classmethod
    def from_canonical_bytes(
        cls,
        payload: bytes,
        *,
        verifier_public_key: bytes,
        runtime_identity: ETTRRuntimeImageIdentity,
        policy: ETTRStagePolicySpec,
        expected_execution_manifest_sha256: str,
    ) -> ETTRStageLaunchReceipt:
        """Strictly parse and authenticate one canonical launch receipt."""

        if not isinstance(payload, bytes) or not 0 < len(payload) <= 1024 * 1024:
            raise ETTRDeploymentContractError(
                "stage launch receipt payload differs"
            )

        def reject_duplicate_keys(
            rows: list[tuple[str, object]],
        ) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in rows:
                if key in value:
                    raise ETTRDeploymentContractError(
                        "stage launch receipt has duplicate keys"
                    )
                value[key] = item
            return value

        try:
            value = json.loads(
                payload.decode("ascii"),
                object_pairs_hook=reject_duplicate_keys,
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ETTRDeploymentContractError,
        ) as exc:
            raise ETTRDeploymentContractError(
                "stage launch receipt is malformed"
            ) from exc
        expected_keys = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ETTRDeploymentContractError(
                "stage launch receipt inventory differs"
            )
        for field in ("input_role_sha256s", "output_role_sha256s"):
            rows = value[field]
            if (
                not isinstance(rows, list)
                or any(
                    not isinstance(row, list)
                    or len(row) != 2
                    or any(not isinstance(item, str) for item in row)
                    for row in rows
                )
            ):
                raise ETTRDeploymentContractError(
                    "stage launch receipt role geometry differs"
                )
            value[field] = tuple(tuple(row) for row in rows)
        try:
            receipt = cls(**value)
        except TypeError as exc:
            raise ETTRDeploymentContractError(
                "stage launch receipt geometry differs"
            ) from exc
        try:
            canonical = receipt.canonical_bytes()
        except (TypeError, ValueError) as exc:
            raise ETTRDeploymentContractError(
                "stage launch receipt geometry differs"
            ) from exc
        if payload != canonical:
            raise ETTRDeploymentContractError(
                "stage launch receipt is not canonical"
            )
        receipt.validate(
            runtime_identity=runtime_identity,
            policy=policy,
            expected_execution_manifest_sha256=(
                expected_execution_manifest_sha256
            ),
            expected_verifier_public_key=verifier_public_key,
        )
        return receipt


def canonical_stage_policy_sha256s() -> dict[Stage, str]:
    return {
        stage: ETTRStagePolicySpec.canonical(stage).sha256()
        for stage in ("world", "command", "query")
    }


def validate_stage_launch_receipt_chain(
    *,
    receipts: Mapping[Stage, ETTRStageLaunchReceipt],
    runtime_identity: ETTRRuntimeImageIdentity,
    expected_execution_manifest_sha256: str,
    expected_verifier_public_key: bytes | None = None,
) -> tuple[str, str, str]:
    """Validate the complete WORLD/COMMAND/QUERY measured-launch chain."""

    if tuple(sorted(receipts)) != ("command", "query", "world"):
        raise ETTRDeploymentContractError(
            "stage launch receipt inventory differs"
        )
    receipt_sha256s: list[str] = []
    for stage in ("world", "command", "query"):
        receipt = receipts[stage]
        receipt.validate(
            runtime_identity=runtime_identity,
            policy=ETTRStagePolicySpec.canonical(stage),
            expected_execution_manifest_sha256=(
                expected_execution_manifest_sha256
            ),
            expected_verifier_public_key=expected_verifier_public_key,
        )
        receipt_sha256s.append(receipt.sha256())
    world, command, query = (
        receipts["world"],
        receipts["command"],
        receipts["query"],
    )
    if (
        command.run_id != world.run_id
        or query.run_id != world.run_id
        or world.parent_launch_receipt_sha256 is not None
        or command.parent_launch_receipt_sha256 != world.sha256()
        or query.parent_launch_receipt_sha256 != command.sha256()
    ):
        raise ETTRDeploymentContractError(
            "stage launch receipt lineage differs"
        )
    return tuple(receipt_sha256s)


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
    "STAGE_LAUNCH_SIGNATURE_DOMAIN",
    "STAGE_POLICY_SCHEMA",
    "canonical_loaded_object_map_sha256",
    "canonical_stage_environment",
    "canonical_stage_environment_sha256",
    "canonical_stage_policy_sha256s",
    "ed25519_public_key_from_private_bytes",
    "ed25519_sign",
    "ed25519_verify",
    "validate_stage_launch_receipt_chain",
]
