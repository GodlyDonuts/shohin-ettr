"""Integrated source compiler using joint assignment-semantics equilibrium.

The compiler retains the maximum witness encoder and replaces the sequential
physical-key controller, same-evidence claim calibrator, categorical revision,
and source-reentry branches with one tied joint equilibrium. Only the final
hard machine and copied opaque keys may cross the seal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import secrets

import torch
import torch.nn as nn

from episode_functor_conflict_compiler import (
    DirectEvidenceProjector,
    record_features_from_witness,
)
from episode_functor_constrained_transport import hard_assign_keys
from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_detached_query_package import module_state_sha256
from episode_functor_joint_equilibrium import (
    JointAssignmentSemanticsEquilibrium,
    JointEquilibriumResult,
)
from episode_functor_machine import HardFunctorKeys, HardFunctorMachine
from episode_functor_witness_compiler import (
    ProofCarryingWitnessCompiler,
    WitnessCompilerBatch,
    WitnessCompilerOutput,
    assemble_relation_evidence,
    canonicalize_witness_batch,
)


class JointCompilerError(ValueError):
    """Integrated joint compiler input, output, or seal failed closed."""


COMPILATION_RECEIPT_SCHEMA = "shohin.efc.jasec-compilation.v1"
SEAL_RECEIPT_SCHEMA = "shohin.efc.jasec-seal.v1"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: object) -> str:
    return sha256(
        (
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")
    ).hexdigest()


def _tensor_bundle_sha256(
    tensors: tuple[tuple[str, torch.Tensor], ...],
) -> str:
    digest = sha256()
    for name, tensor in tensors:
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(tensor, torch.Tensor)
        ):
            raise JointCompilerError("compilation tensor bundle differs")
        contiguous = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(contiguous.dtype).encode("ascii")
        digest.update(len(name_bytes).to_bytes(4, "little"))
        digest.update(name_bytes)
        digest.update(len(dtype_bytes).to_bytes(4, "little"))
        digest.update(dtype_bytes)
        digest.update(contiguous.ndim.to_bytes(4, "little"))
        for dimension in contiguous.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class JointCompilationReceipt:
    schema: str
    compiler_instance_nonce: str
    compiler_state_sha256: str
    compiler_configuration_sha256: str
    compiler_parameter_count: int
    source_sha256: tuple[str, ...]
    key_inventory_sha256: str
    equilibrium_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema != COMPILATION_RECEIPT_SCHEMA
            or not _is_sha256(self.compiler_instance_nonce)
            or not _is_sha256(self.compiler_state_sha256)
            or not _is_sha256(self.compiler_configuration_sha256)
            or not isinstance(self.compiler_parameter_count, int)
            or isinstance(self.compiler_parameter_count, bool)
            or self.compiler_parameter_count < 1
            or not self.source_sha256
            or any(not _is_sha256(value) for value in self.source_sha256)
            or not _is_sha256(self.key_inventory_sha256)
            or not _is_sha256(self.equilibrium_sha256)
        ):
            raise JointCompilerError(
                "joint compilation receipt differs"
            )

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class JointCompilerOutput:
    """Attached source state; equilibrium diagnostics must not cross the seal."""

    initial_witness: WitnessCompilerOutput
    equilibrium: JointEquilibriumResult
    witness: WitnessCompilerOutput
    receipt: JointCompilationReceipt
    compiler_capability: object

    def __post_init__(self) -> None:
        if (
            self.initial_witness.raw_key_assignment_logits.shape
            != self.equilibrium.raw_key_assignment_logits.shape
            or self.witness.raw_key_assignment_logits.shape
            != self.equilibrium.raw_key_assignment_logits.shape
            or self.witness.key_assignment_logits.shape
            != self.equilibrium.key_assignment_logits.shape
            or self.witness.projection.transition_transport.shape
            != self.equilibrium.transition_probabilities.shape
            or self.witness.projection.observer_transport.shape
            != self.equilibrium.observer_probabilities.shape
            or self.initial_witness.source_sha256
            != self.witness.source_sha256
            or self.receipt.source_sha256
            != self.witness.source_sha256
            or type(self.compiler_capability) is not object
        ):
            raise JointCompilerError(
                "integrated joint compiler output differs"
            )


@dataclass(frozen=True, slots=True)
class SealedJointMachine:
    machine: HardFunctorMachine
    keys: HardFunctorKeys
    wire_sha256: tuple[str, ...]
    seal_receipt_sha256: str
    seal_capability: object

    def __post_init__(self) -> None:
        if (
            self.machine.batch_size != self.keys.batch_size
            or len(self.wire_sha256) != self.machine.batch_size
            or any(not _is_sha256(value) for value in self.wire_sha256)
            or not _is_sha256(self.seal_receipt_sha256)
            or type(self.seal_capability) is not object
        ):
            raise JointCompilerError(
                "sealed joint machine and key batches differ"
            )
        expected = (
            (self.machine.state_active, PRIMARY_STATES),
            (self.machine.action_active, PRIMARY_ACTIONS),
            (self.machine.observer_active, PRIMARY_OBSERVERS),
        )
        for active, count in expected:
            if (
                not bool(active[:, :count].eq(1).all())
                or bool(active[:, count:].ne(0).any())
            ):
                raise JointCompilerError(
                    "sealed joint machine leaves primary geometry"
                )
        for row in range(self.machine.batch_size):
            self.keys.validate_masks(self.machine, row)

    def deployed_wire(self, row: int) -> bytes:
        payload = self.machine.deployed_wire(self.keys, row)
        if sha256(payload).hexdigest() != self.wire_sha256[row]:
            raise JointCompilerError(
                "sealed joint wire differs from its receipt"
            )
        return payload


@dataclass(frozen=True, slots=True)
class _IssuedSealReceipt:
    compiler_state_sha256: str
    compiler_configuration_sha256: str
    compiler_parameter_count: int
    wire_sha256: tuple[str, ...]
    seal_receipt_sha256: str


class JointProofCarryingCompiler(nn.Module):
    """Maximum anonymous-source encoder plus tied joint equilibrium."""

    def __init__(
        self,
        *,
        external_feature_width: int = 1728,
        width: int = 512,
        encoder_layers: int = 8,
        decoder_layers: int = 4,
        heads: int = 16,
        feedforward: int = 2048,
        assignment_width: int = 600,
        assignment_context_width: int = 1200,
        machine_width: int = 960,
        machine_context_width: int = 1920,
        cycles: int = 4,
        sinkhorn_iterations: int = 64,
    ) -> None:
        super().__init__()
        self.witness = ProofCarryingWitnessCompiler(
            width=width,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            heads=heads,
            feedforward=feedforward,
            external_feature_width=external_feature_width,
            sinkhorn_iterations=sinkhorn_iterations,
            opaque_key_invariant=True,
            projector=DirectEvidenceProjector(),
        )
        self.equilibrium = JointAssignmentSemanticsEquilibrium(
            assignment_width=assignment_width,
            assignment_context_width=assignment_context_width,
            machine_width=machine_width,
            machine_context_width=machine_context_width,
            cycles=cycles,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        self._instance_nonce = secrets.token_hex(32)
        self._compiler_capability = object()
        self._issued_seals: dict[object, _IssuedSealReceipt] = {}

    @property
    def external_feature_width(self) -> int:
        return self.witness.external_feature_width

    @property
    def projector(self) -> DirectEvidenceProjector:
        projector = self.witness.projector
        if type(projector) is not DirectEvidenceProjector:
            raise JointCompilerError(
                "joint compiler projector differs"
            )
        return projector

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def configuration_sha256(self) -> str:
        """Bind every mutable non-tensor architecture setting."""

        encoder_layer = self.witness.encoder.layers[0]
        decoder_layer = self.witness.slot_decoder.layers[0]
        return _canonical_sha256(
            {
                "equilibrium": {
                    "assignment_context_width": (
                        self.equilibrium.assignment_context_width
                    ),
                    "assignment_feature_width": (
                        self.equilibrium.assignment_feature_width
                    ),
                    "assignment_width": (
                        self.equilibrium.assignment_width
                    ),
                    "cycles": self.equilibrium.cycles,
                    "machine_context_width": (
                        self.equilibrium.machine_context_width
                    ),
                    "machine_feature_width": (
                        self.equilibrium.machine_feature_width
                    ),
                    "machine_width": self.equilibrium.machine_width,
                    "max_assignment_correction": (
                        self.equilibrium.max_assignment_correction
                    ),
                    "max_machine_step": (
                        self.equilibrium.max_machine_step
                    ),
                    "sinkhorn_iterations": (
                        self.equilibrium.sinkhorn_iterations
                    ),
                },
                "projector_type": type(self.projector).__qualname__,
                "schema": "shohin.efc.jasec-configuration.v1",
                "witness": {
                    "decoder_feedforward": (
                        decoder_layer.linear1.out_features
                    ),
                    "decoder_heads": (
                        decoder_layer.self_attn.num_heads
                    ),
                    "decoder_layers": len(
                        self.witness.slot_decoder.layers
                    ),
                    "encoder_feedforward": (
                        encoder_layer.linear1.out_features
                    ),
                    "encoder_heads": (
                        encoder_layer.self_attn.num_heads
                    ),
                    "encoder_layers": len(self.witness.encoder.layers),
                    "external_feature_width": (
                        self.witness.external_feature_width
                    ),
                    "opaque_key_invariant": (
                        self.witness.opaque_key_invariant
                    ),
                    "sinkhorn_iterations": (
                        self.witness.key_sinkhorn_iterations
                    ),
                    "width": self.witness.width,
                },
            }
        )

    def deployment_sha256(self) -> str:
        """Bind tensor state and non-tensor execution configuration."""

        return _canonical_sha256(
            {
                "configuration_sha256": self.configuration_sha256(),
                "parameter_count": self.parameter_count(),
                "schema": "shohin.efc.jasec-deployment.v1",
                "state_sha256": module_state_sha256(self),
            }
        )

    @staticmethod
    def _key_inventory_sha256(
        witness: WitnessCompilerOutput,
    ) -> str:
        return _tensor_bundle_sha256(
            (
                ("unique_key_bytes", witness.unique_key_bytes),
                ("unique_key_valid", witness.unique_key_valid),
            )
        )

    @staticmethod
    def _equilibrium_sha256(
        equilibrium: JointEquilibriumResult,
    ) -> str:
        return _tensor_bundle_sha256(
            (
                (
                    "raw_key_assignment_logits",
                    equilibrium.raw_key_assignment_logits,
                ),
                (
                    "key_assignment_logits",
                    equilibrium.key_assignment_logits,
                ),
                (
                    "transition_probabilities",
                    equilibrium.transition_probabilities,
                ),
                (
                    "observer_probabilities",
                    equilibrium.observer_probabilities,
                ),
            )
        )

    @staticmethod
    def _relation_evidence(
        batch: WitnessCompilerBatch,
        witness: WitnessCompilerOutput,
        key_assignment_logits: torch.Tensor,
    ):
        return assemble_relation_evidence(
            record_type_logits=witness.record_type_logits,
            occurrence_role_logits=witness.occurrence_role_logits,
            answer_logits=witness.answer_logits,
            occurrence_valid=batch.pointer.occurrence_valid,
            occurrence_to_record=batch.occurrence_to_record,
            occurrence_to_unique=batch.pointer.occurrence_to_unique,
            source_unique_key_valid=witness.unique_key_valid,
            key_assignment_logits=key_assignment_logits,
        )

    def forward(
        self,
        batch: WitnessCompilerBatch,
        *,
        straight_through: bool = False,
        frozen_byte_features: torch.Tensor | None = None,
        mode: str = "causal",
    ) -> JointCompilerOutput:
        if straight_through:
            raise JointCompilerError(
                "joint compiler forbids solver-backed training assignment"
            )
        model_batch = canonicalize_witness_batch(batch)
        initial = self.witness(
            batch,
            straight_through=False,
            frozen_byte_features=frozen_byte_features,
        )
        equilibrium = self.equilibrium(
            model_batch,
            initial,
            record_features=record_features_from_witness(initial),
            mode=mode,
        )
        relation_evidence = self._relation_evidence(
            model_batch,
            initial,
            equilibrium.key_assignment_logits,
        )
        tiny = torch.finfo(
            equilibrium.transition_probabilities.dtype
        ).tiny
        projection = self.projector(
            equilibrium.transition_probabilities.clamp_min(tiny).log(),
            equilibrium.observer_probabilities.clamp_min(tiny).log(),
            straight_through=False,
        )
        witness = replace(
            initial,
            projection=projection,
            relation_evidence=relation_evidence,
            key_assignment_logits=equilibrium.key_assignment_logits,
            raw_key_assignment_logits=(
                equilibrium.raw_key_assignment_logits
            ),
            projector_auxiliary=None,
        )
        receipt = JointCompilationReceipt(
            schema=COMPILATION_RECEIPT_SCHEMA,
            compiler_instance_nonce=self._instance_nonce,
            compiler_state_sha256=module_state_sha256(self),
            compiler_configuration_sha256=self.configuration_sha256(),
            compiler_parameter_count=self.parameter_count(),
            source_sha256=witness.source_sha256,
            key_inventory_sha256=self._key_inventory_sha256(witness),
            equilibrium_sha256=self._equilibrium_sha256(equilibrium),
        )
        return JointCompilerOutput(
            initial_witness=initial,
            equilibrium=equilibrium,
            witness=witness,
            receipt=receipt,
            compiler_capability=self._compiler_capability,
        )

    @torch.no_grad()
    def seal(
        self,
        output: JointCompilerOutput,
    ) -> SealedJointMachine:
        if type(output) is not JointCompilerOutput:
            raise JointCompilerError(
                "joint compiler may seal only its own output"
            )
        receipt = output.receipt
        if (
            type(receipt) is not JointCompilationReceipt
            or output.compiler_capability is not self._compiler_capability
            or receipt.compiler_instance_nonce != self._instance_nonce
            or receipt.compiler_state_sha256 != module_state_sha256(self)
            or receipt.compiler_configuration_sha256
            != self.configuration_sha256()
            or receipt.compiler_parameter_count != self.parameter_count()
            or receipt.source_sha256
            != output.initial_witness.source_sha256
            or receipt.source_sha256 != output.witness.source_sha256
            or receipt.key_inventory_sha256
            != self._key_inventory_sha256(output.witness)
            or receipt.key_inventory_sha256
            != self._key_inventory_sha256(output.initial_witness)
            or receipt.equilibrium_sha256
            != self._equilibrium_sha256(output.equilibrium)
        ):
            raise JointCompilerError(
                "joint compilation provenance differs"
            )
        tiny = torch.finfo(
            output.equilibrium.transition_probabilities.dtype
        ).tiny
        machine = self.projector.hard_project(
            output.equilibrium.transition_probabilities.clamp_min(tiny).log(),
            output.equilibrium.observer_probabilities.clamp_min(tiny).log(),
        )
        key_projection = hard_assign_keys(
            slot_assignment_logits=(
                output.equilibrium.raw_key_assignment_logits
            ),
            source_unique_key_bytes=output.witness.unique_key_bytes,
            source_unique_key_valid=output.witness.unique_key_valid,
        )
        wires = tuple(
            machine.deployed_wire(key_projection.keys, row)
            for row in range(machine.batch_size)
        )
        wire_sha256 = tuple(
            sha256(payload).hexdigest() for payload in wires
        )
        seal_receipt_sha256 = _canonical_sha256(
            {
                "compilation_receipt_sha256": receipt.receipt_sha256,
                "schema": SEAL_RECEIPT_SCHEMA,
                "wire_sha256": wire_sha256,
            }
        )
        seal_capability = object()
        sealed = SealedJointMachine(
            machine=machine,
            keys=key_projection.keys,
            wire_sha256=wire_sha256,
            seal_receipt_sha256=seal_receipt_sha256,
            seal_capability=seal_capability,
        )
        self._issued_seals[seal_capability] = _IssuedSealReceipt(
            compiler_state_sha256=receipt.compiler_state_sha256,
            compiler_configuration_sha256=(
                receipt.compiler_configuration_sha256
            ),
            compiler_parameter_count=receipt.compiler_parameter_count,
            wire_sha256=wire_sha256,
            seal_receipt_sha256=seal_receipt_sha256,
        )
        return sealed

    @torch.no_grad()
    def verify_sealed(
        self,
        sealed: SealedJointMachine,
    ) -> tuple[bytes, ...]:
        """Revalidate an issued seal against this exact compiler state."""

        if type(sealed) is not SealedJointMachine:
            raise JointCompilerError(
                "joint seal provenance type differs"
            )
        issued = self._issued_seals.get(sealed.seal_capability)
        if (
            issued is None
            or issued.compiler_state_sha256 != module_state_sha256(self)
            or issued.compiler_configuration_sha256
            != self.configuration_sha256()
            or issued.compiler_parameter_count != self.parameter_count()
        ):
            raise JointCompilerError(
                "joint seal compiler provenance differs"
            )
        wires = tuple(
            sealed.deployed_wire(row)
            for row in range(sealed.machine.batch_size)
        )
        expected_wire_sha256 = tuple(
            sha256(payload).hexdigest() for payload in wires
        )
        if (
            sealed.wire_sha256 != expected_wire_sha256
            or sealed.wire_sha256 != issued.wire_sha256
            or sealed.seal_receipt_sha256
            != issued.seal_receipt_sha256
        ):
            raise JointCompilerError(
                "joint seal receipt differs"
            )
        return wires


__all__ = [
    "JointCompilerError",
    "JointCompilerOutput",
    "JointCompilationReceipt",
    "JointProofCarryingCompiler",
    "COMPILATION_RECEIPT_SCHEMA",
    "SEAL_RECEIPT_SCHEMA",
    "SealedJointMachine",
]
