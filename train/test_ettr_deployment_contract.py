from __future__ import annotations

from dataclasses import replace

import pytest

from ettr_deployment_contract import (
    ETTRDeploymentContractError,
    ETTRRuntimeImageIdentity,
    ETTRStageLaunchReceipt,
    ETTRStagePolicySpec,
    RUNTIME_IDENTITY_SCHEMA,
    STAGE_LAUNCH_RECEIPT_SCHEMA,
    canonical_stage_policy_sha256s,
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
    receipt = ETTRStageLaunchReceipt(
        schema=STAGE_LAUNCH_RECEIPT_SCHEMA,
        stage=stage,
        execution_manifest_sha256="2" * 64,
        runtime_identity_sha256=identity.sha256(),
        stage_policy_sha256=policy.sha256(),
        bwrap_sha256=identity.bwrap_sha256,
        input_role_sha256s=tuple(
            (role, "3" * 64) for role in policy.read_roles
        ),
        output_role_sha256s=tuple(
            (role, "4" * 64) for role in policy.write_roles
        ),
        parent_network_namespace="1:2",
        child_network_namespace="1:3",
        environment_sha256="5" * 64,
        loaded_object_map_sha256="6" * 64,
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
