from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from ettr_deployment_contract import (
    ETTRDeploymentContractError,
    ETTRRuntimeImageIdentity,
    ETTRStageLaunchReceipt,
    ETTRStagePolicySpec,
    RUNTIME_IDENTITY_SCHEMA,
    STAGE_LAUNCH_RECEIPT_SCHEMA,
    STAGE_LAUNCH_SIGNATURE_DOMAIN,
    canonical_loaded_object_map_sha256,
    canonical_stage_environment_sha256,
    canonical_stage_policy_sha256s,
    validate_stage_launch_receipt_chain,
)


_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"k" * 32)
_PUBLIC_KEY = _PRIVATE_KEY.public_key().public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw,
)


def _identity() -> ETTRRuntimeImageIdentity:
    return ETTRRuntimeImageIdentity(
        schema=RUNTIME_IDENTITY_SCHEMA,
        archive_sha256="a" * 64,
        archive_size=123,
        inventory_sha256="b" * 64,
        world_runtime_bundle_sha256="c" * 64,
        command_runtime_bundle_sha256="7" * 64,
        query_runtime_bundle_sha256="8" * 64,
        python_sha256="d" * 64,
        bootstrap_sha256="e" * 64,
        external_launcher_sha256="f" * 64,
        bwrap_sha256="1" * 64,
        network_namespace_required=True,
    )


def _launch(
    stage: str,
) -> tuple[
    ETTRStageLaunchReceipt,
    ETTRRuntimeImageIdentity,
    ETTRStagePolicySpec,
]:
    identity = _identity()
    policy = ETTRStagePolicySpec.canonical(stage)
    input_roles = tuple(
        (role, "3" * 64) for role in policy.read_roles
    )
    output_roles = tuple(
        (role, "4" * 64) for role in policy.write_roles
    )
    unsigned = ETTRStageLaunchReceipt(
        schema=STAGE_LAUNCH_RECEIPT_SCHEMA,
        stage=stage,
        run_id="9" * 64,
        parent_launch_receipt_sha256=(
            None if stage == "world" else "8" * 64
        ),
        verifier_public_key_sha256=hashlib.sha256(_PUBLIC_KEY).hexdigest(),
        execution_manifest_sha256="2" * 64,
        runtime_identity_sha256=identity.sha256(),
        stage_policy_sha256=policy.sha256(),
        bwrap_sha256=identity.bwrap_sha256,
        input_role_sha256s=input_roles,
        output_role_sha256s=output_roles,
        parent_network_namespace="1:2",
        child_network_namespace="1:3",
        allocated_gpu_minor=0,
        exit_code=0,
        stdout_sha256="5" * 64,
        stderr_sha256="6" * 64,
        environment_sha256=canonical_stage_environment_sha256(
            stage=stage,
            runtime_identity=identity,
            allocated_gpu_minor=0,
        ),
        loaded_object_map_sha256=canonical_loaded_object_map_sha256(
            stage=stage,
            runtime_identity=identity,
            input_role_sha256s=input_roles,
            output_role_sha256s=output_roles,
        ),
        verifier_signature_hex="",
    )
    receipt = replace(
        unsigned,
        verifier_signature_hex=_PRIVATE_KEY.sign(
            unsigned.signing_bytes()
        ).hex(),
    )
    return receipt, identity, policy


@pytest.mark.parametrize("stage", ["world", "command", "query"])
def test_canonical_stage_policies_and_launch_receipts_validate(
    stage: str,
) -> None:
    receipt, identity, policy = _launch(stage)
    identity.validate()
    policy.validate()
    receipt.validate(
        runtime_identity=identity,
        policy=policy,
        expected_execution_manifest_sha256="2" * 64,
        expected_verifier_public_key=_PUBLIC_KEY,
    )
    assert canonical_stage_policy_sha256s()[stage] == policy.sha256()


def test_runtime_identity_reconstructs_exact_manifest_bindings() -> None:
    identity = _identity()
    manifest = {
        "bootstrap_sha256": identity.bootstrap_sha256,
        "bwrap_sha256": identity.bwrap_sha256,
        "claim_runtime_archive_sha256": identity.archive_sha256,
        "claim_runtime_archive_size": identity.archive_size,
        "claim_runtime_inventory_sha256": identity.inventory_sha256,
        "external_launcher_sha256": identity.external_launcher_sha256,
        "network_namespace_required": True,
        "world_runtime_bundle_sha256": (
            identity.world_runtime_bundle_sha256
        ),
        "command_runtime_bundle_sha256": (
            identity.command_runtime_bundle_sha256
        ),
        "query_runtime_bundle_sha256": (
            identity.query_runtime_bundle_sha256
        ),
    }
    assert ETTRRuntimeImageIdentity.from_manifest(
        manifest,
        python_sha256=identity.python_sha256,
    ) == identity


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("read_roles", ("runtime_image",)),
        ("write_roles", ("attacker_output",)),
        ("deny_network", False),
        ("unshare_pid", False),
        ("forbid_repository_root", False),
        ("forbid_assessor", False),
        ("forbid_signing_keys", False),
    ],
)
def test_stage_policy_widening_fails_closed(field: str, value: object) -> None:
    policy = ETTRStagePolicySpec.canonical("world")
    with pytest.raises(ETTRDeploymentContractError, match="policy differs"):
        replace(policy, **{field: value}).validate()


def test_launch_rejects_role_reassociation_and_extra_role() -> None:
    receipt, identity, policy = _launch("command")
    swapped = list(receipt.input_role_sha256s)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    extra = (*receipt.input_role_sha256s, ("assessor", "7" * 64))
    for input_roles in (tuple(swapped), extra):
        with pytest.raises(
            ETTRDeploymentContractError,
            match="role inventory differs",
        ):
            replace(
                receipt,
                input_role_sha256s=input_roles,
            ).validate(
                runtime_identity=identity,
                policy=policy,
                expected_execution_manifest_sha256="2" * 64,
            )


def test_launch_rejects_same_network_namespace_and_runtime_substitution() -> None:
    receipt, identity, policy = _launch("query")
    with pytest.raises(
        ETTRDeploymentContractError,
        match="launch receipt differs",
    ):
        replace(
            receipt,
            child_network_namespace=receipt.parent_network_namespace,
        ).validate(
            runtime_identity=identity,
            policy=policy,
            expected_execution_manifest_sha256="2" * 64,
        )
    with pytest.raises(
        ETTRDeploymentContractError,
        match="launch receipt differs",
    ):
        receipt.validate(
            runtime_identity=replace(
                identity,
                archive_sha256="8" * 64,
            ),
            policy=policy,
            expected_execution_manifest_sha256="2" * 64,
        )


def test_runtime_identity_rejects_unmeasured_or_nonisolated_runtime() -> None:
    identity = _identity()
    for changed in (
        replace(identity, archive_sha256="not-a-hash"),
        replace(identity, archive_size=0),
        replace(identity, network_namespace_required=False),
    ):
        with pytest.raises(ETTRDeploymentContractError):
            changed.validate()


def test_measured_environment_object_map_and_exit_are_not_assertions() -> None:
    receipt, identity, policy = _launch("world")
    for changed in (
        replace(receipt, environment_sha256="0" * 64),
        replace(receipt, loaded_object_map_sha256="0" * 64),
        replace(receipt, allocated_gpu_minor=1),
        replace(receipt, exit_code=1),
    ):
        with pytest.raises(ETTRDeploymentContractError):
            changed.validate(
                runtime_identity=identity,
                policy=policy,
                expected_execution_manifest_sha256="2" * 64,
            )


def test_complete_launch_chain_rejects_missing_or_reassociated_stage() -> None:
    world = _launch("world")[0]
    command_unsigned = replace(
        _launch("command")[0],
        parent_launch_receipt_sha256=world.sha256(),
        verifier_signature_hex="",
    )
    command = replace(
        command_unsigned,
        verifier_signature_hex=_PRIVATE_KEY.sign(
            command_unsigned.signing_bytes()
        ).hex(),
    )
    query_unsigned = replace(
        _launch("query")[0],
        parent_launch_receipt_sha256=command.sha256(),
        verifier_signature_hex="",
    )
    query = replace(
        query_unsigned,
        verifier_signature_hex=_PRIVATE_KEY.sign(
            query_unsigned.signing_bytes()
        ).hex(),
    )
    receipts = {"world": world, "command": command, "query": query}
    identity = _identity()
    assert len(
        validate_stage_launch_receipt_chain(
            receipts=receipts,
            runtime_identity=identity,
            expected_execution_manifest_sha256="2" * 64,
            expected_verifier_public_key=_PUBLIC_KEY,
        )
    ) == 3
    with pytest.raises(
        ETTRDeploymentContractError,
        match="inventory differs",
    ):
        validate_stage_launch_receipt_chain(
            receipts={
                "world": receipts["world"],
                "command": receipts["command"],
            },
            runtime_identity=identity,
            expected_execution_manifest_sha256="2" * 64,
            expected_verifier_public_key=_PUBLIC_KEY,
        )
    with pytest.raises(ETTRDeploymentContractError):
        validate_stage_launch_receipt_chain(
            receipts={
                **receipts,
                "query": replace(receipts["query"], stage="world"),
            },
            runtime_identity=identity,
            expected_execution_manifest_sha256="2" * 64,
            expected_verifier_public_key=_PUBLIC_KEY,
        )
    broken_query_unsigned = replace(
        receipts["query"],
        parent_launch_receipt_sha256="8" * 64,
        verifier_signature_hex="",
    )
    broken_query = replace(
        broken_query_unsigned,
        verifier_signature_hex=_PRIVATE_KEY.sign(
            broken_query_unsigned.signing_bytes()
        ).hex(),
    )
    with pytest.raises(
        ETTRDeploymentContractError,
        match="lineage differs",
    ):
        validate_stage_launch_receipt_chain(
            receipts={**receipts, "query": broken_query},
            runtime_identity=identity,
            expected_execution_manifest_sha256="2" * 64,
            expected_verifier_public_key=_PUBLIC_KEY,
        )


def test_launch_receipt_signature_is_domain_separated_and_tamper_evident() -> None:
    receipt, identity, policy = _launch("command")
    with pytest.raises(InvalidSignature):
        _PRIVATE_KEY.public_key().verify(
            bytes.fromhex(receipt.verifier_signature_hex),
            receipt.signing_bytes().removeprefix(
                STAGE_LAUNCH_SIGNATURE_DOMAIN
            ),
        )
    for changed in (
        replace(receipt, stdout_sha256="0" * 64),
        replace(receipt, run_id="7" * 64),
        replace(receipt, parent_launch_receipt_sha256="6" * 64),
        replace(receipt, verifier_signature_hex="0" * 128),
    ):
        with pytest.raises(ETTRDeploymentContractError):
            changed.validate(
                runtime_identity=identity,
                policy=policy,
                expected_execution_manifest_sha256="2" * 64,
                expected_verifier_public_key=_PUBLIC_KEY,
            )
    wrong_public_key = (
        Ed25519PrivateKey.from_private_bytes(b"x" * 32)
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )
    with pytest.raises(
        ETTRDeploymentContractError,
        match="public key identity differs",
    ):
        receipt.validate(
            runtime_identity=identity,
            policy=policy,
            expected_execution_manifest_sha256="2" * 64,
            expected_verifier_public_key=wrong_public_key,
        )


@pytest.mark.parametrize("stage", ["world", "command", "query"])
def test_launch_receipt_parent_geometry_is_stage_specific(stage: str) -> None:
    receipt, identity, policy = _launch(stage)
    invalid_parent = "7" * 64 if stage == "world" else None
    with pytest.raises(ETTRDeploymentContractError):
        replace(
            receipt,
            parent_launch_receipt_sha256=invalid_parent,
        ).validate(
            runtime_identity=identity,
            policy=policy,
            expected_execution_manifest_sha256="2" * 64,
            expected_verifier_public_key=_PUBLIC_KEY,
        )


def test_strict_launch_receipt_parser_authenticates_canonical_bytes() -> None:
    receipt, identity, policy = _launch("query")
    assert ETTRStageLaunchReceipt.from_canonical_bytes(
        receipt.canonical_bytes(),
        verifier_public_key=_PUBLIC_KEY,
        runtime_identity=identity,
        policy=policy,
        expected_execution_manifest_sha256="2" * 64,
    ) == receipt

    pretty = json.dumps(
        json.loads(receipt.canonical_bytes()),
        indent=2,
        sort_keys=True,
    ).encode("ascii")
    duplicate = receipt.canonical_bytes().replace(
        b'{"allocated_gpu_minor":',
        b'{"allocated_gpu_minor":0,"allocated_gpu_minor":',
        1,
    )
    extra = {
        **json.loads(receipt.canonical_bytes()),
        "unexpected": True,
    }
    tampered = replace(
        receipt,
        verifier_signature_hex="0" * 128,
    ).canonical_bytes()
    for payload in (
        pretty,
        duplicate,
        json.dumps(extra, sort_keys=True).encode("ascii"),
        tampered,
    ):
        with pytest.raises(ETTRDeploymentContractError):
            ETTRStageLaunchReceipt.from_canonical_bytes(
                payload,
                verifier_public_key=_PUBLIC_KEY,
                runtime_identity=identity,
                policy=policy,
                expected_execution_manifest_sha256="2" * 64,
            )
