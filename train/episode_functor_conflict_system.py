"""Connected system and supervised objective for conflict-reentrant EFC.

The wrapper preserves the established frozen-Shohin and detached-query
interfaces while replacing the one-pass source compiler with the tied
two-pass conflict compiler. Only the second-pass hard machine and copied keys
cross the seal.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Sequence

import torch

from pipeline.episode_functor_qualification_batch import (
    QualificationSupervisorBatch,
)
from episode_functor_conflict_compiler import (
    ConflictCompilerOutput,
    ConflictProofCarryingCompiler,
)
from episode_functor_detached_query_package import (
    DetachedExecutionAuthorization,
    DetachedQueryParserReceipt,
    PACKAGE_SCHEMA,
    build_detached_execution_authorization,
    detached_query_parser_state_sha256,
    module_state_sha256,
)
from episode_functor_learned_system import (
    EFCParameterReceipt,
    GLOBAL_PARAMETER_LIMIT,
    LearnedEFCSystem,
    PROTECTED_SHOHIN_SHA256,
    SealedFunctorBatch,
)
from episode_functor_qualification_loss import (
    EFCQualificationLoss,
    QualificationExactMetrics,
    QualificationLossOutput,
    QualificationLossWeights,
)
from episode_functor_query_parser import NeuralOpaqueQueryParser
from episode_functor_shohin_trunk import (
    FrozenShohinTrunk,
    ShohinTrunkBatch,
)
from episode_functor_witness_compiler import (
    WitnessCompilerBatch,
    WitnessCompilerOutput,
)

PROTECTED_SHOHIN_PARAMETERS = 125_081_664
MAXIMUM_SYSTEM_PARAMETERS = 200_000_000


class ConflictSystemError(ValueError):
    """Conflict-system compilation, supervision, or seal failed closed."""


class ConflictReentrantEFCSystem(LearnedEFCSystem):
    """Frozen Shohin plus tied two-pass source compiler and late query parser."""

    source_compiler: ConflictProofCarryingCompiler

    def __init__(
        self,
        *,
        source_compiler: ConflictProofCarryingCompiler | None = None,
        query_parser: NeuralOpaqueQueryParser | None = None,
        query_parser_receipt: DetachedQueryParserReceipt | None = None,
        frozen_trunk: FrozenShohinTrunk | None = None,
    ) -> None:
        if frozen_trunk is None:
            raise ConflictSystemError(
                "conflict system requires a frozen Shohin trunk"
            )
        if not isinstance(frozen_trunk, FrozenShohinTrunk):
            raise ConflictSystemError(
                "conflict system requires FrozenShohinTrunk custody"
            )
        trunk_receipt = frozen_trunk.parameter_receipt()
        if (
            trunk_receipt.parent_unique_parameters
            != PROTECTED_SHOHIN_PARAMETERS
            or trunk_receipt.adapter_unique_parameters != 0
            or trunk_receipt.integrated_unique_parameters
            != PROTECTED_SHOHIN_PARAMETERS
        ):
            raise ConflictSystemError(
                "conflict system requires the exact adapter-free Shohin trunk"
            )
        external_width = int(frozen_trunk.feature_width)
        compiler = (
            ConflictProofCarryingCompiler(
                external_feature_width=external_width,
            )
            if source_compiler is None
            else source_compiler
        )
        if type(compiler) is not ConflictProofCarryingCompiler:
            raise ConflictSystemError(
                "conflict system requires the conflict source compiler"
            )
        source_compiler_parameters = sum(
            parameter.numel() for parameter in compiler.parameters()
        )
        if query_parser is None or query_parser_receipt is None:
            raise ConflictSystemError(
                "conflict system requires a preregistered detached parser"
            )
        parser = query_parser
        if type(parser) is not NeuralOpaqueQueryParser:
            raise ConflictSystemError(
                "detached query parser must have the exact approved type"
            )
        if parser.external_feature_width != 0:
            raise ConflictSystemError(
                "detached query parser must be source-independent"
            )
        detached_query_parameters = sum(
            parameter.numel() for parameter in parser.parameters()
        )
        parser_architecture = {
            "external_feature_width": parser.external_feature_width,
            "feedforward": parser.feedforward,
            "heads": parser.heads,
            "layers": parser.layers,
            "max_steps": parser.max_steps,
            "width": parser.width,
        }
        parser_state_sha256 = detached_query_parser_state_sha256(parser)
        if (
            query_parser_receipt.schema != PACKAGE_SCHEMA
            or query_parser_receipt.parameter_count
            != detached_query_parameters
            or query_parser_receipt.architecture != parser_architecture
            or query_parser_receipt.state_sha256 != parser_state_sha256
        ):
            raise ConflictSystemError(
                "detached parser differs from its preregistered package"
            )
        # The parent wrapper couples source and query feature widths because
        # its parser is same-process. This system deliberately splits those
        # phases: only the compiler receives frozen source residuals.
        torch.nn.Module.__init__(self)
        self.source_compiler = compiler
        self.frozen_trunk = frozen_trunk
        complete_parameters = (
            trunk_receipt.integrated_unique_parameters
            + source_compiler_parameters
            + detached_query_parameters
        )
        if complete_parameters >= MAXIMUM_SYSTEM_PARAMETERS:
            raise ConflictSystemError(
                "conflict system reaches or exceeds the 200M limit"
            )
        self._detached_query_parameters = detached_query_parameters
        self._source_compiler_parameters = source_compiler_parameters
        self._detached_query_architecture = parser_architecture
        self._detached_query_state_sha256 = parser_state_sha256
        self._detached_query_manifest_sha256 = (
            query_parser_receipt.manifest_sha256
        )
        self._detached_query_weights_sha256 = (
            query_parser_receipt.weights_sha256
        )
        self._detached_query_receipt = query_parser_receipt

    def added_parameter_count(self) -> int:
        return (
            self._source_compiler_parameters
            + self._detached_query_parameters
        )

    def complete_parameter_count(self) -> int:
        return PROTECTED_SHOHIN_PARAMETERS + self.added_parameter_count()

    def parameter_headroom(self) -> int:
        return MAXIMUM_SYSTEM_PARAMETERS - self.complete_parameter_count()

    @property
    def detached_query_parameter_count(self) -> int:
        return self._detached_query_parameters

    @property
    def detached_query_architecture(self) -> dict[str, int]:
        return dict(self._detached_query_architecture)

    @property
    def detached_query_state_sha256(self) -> str:
        return self._detached_query_state_sha256

    @property
    def detached_query_manifest_sha256(self) -> str:
        return self._detached_query_manifest_sha256

    @property
    def detached_query_weights_sha256(self) -> str:
        return self._detached_query_weights_sha256

    def authorize_deployed_wire(
        self,
        wire: bytes,
    ) -> DetachedExecutionAuthorization:
        trunk_receipt = self.frozen_trunk.parameter_receipt()
        if not isinstance(wire, bytes) or len(wire) != 1_536:
            raise ConflictSystemError(
                "deployment authorization requires one exact wire"
            )
        if (
            not trunk_receipt.checkpoint_verified
            or trunk_receipt.checkpoint_sha256 != PROTECTED_SHOHIN_SHA256
        ):
            raise ConflictSystemError(
                "deployment authorization requires verified Shohin custody"
            )
        return build_detached_execution_authorization(
            machine_sha256=sha256(wire).hexdigest(),
            parser_receipt=self._detached_query_receipt,
            source_compiler_parameter_count=self._source_compiler_parameters,
            source_compiler_state_sha256=module_state_sha256(
                self.source_compiler
            ),
        )

    def parameter_receipt(
        self,
        *,
        protected_shohin: int = PROTECTED_SHOHIN_PARAMETERS,
        protected_checkpoint_sha256: str = PROTECTED_SHOHIN_SHA256,
        global_limit: int = GLOBAL_PARAMETER_LIMIT,
    ) -> EFCParameterReceipt:
        trunk_receipt = self.frozen_trunk.parameter_receipt()
        source_parameters = self._source_compiler_parameters
        query_parameters = self._detached_query_parameters
        integrated_shohin = trunk_receipt.integrated_unique_parameters
        connected = (
            integrated_shohin == protected_shohin
            and trunk_receipt.checkpoint_verified
            and trunk_receipt.checkpoint_sha256
            == protected_checkpoint_sha256
        )
        return EFCParameterReceipt(
            protected_shohin_reference=protected_shohin,
            protected_checkpoint_sha256=protected_checkpoint_sha256,
            integrated_shohin=integrated_shohin,
            integrated_checkpoint_sha256=trunk_receipt.checkpoint_sha256,
            checkpoint_verified=trunk_receipt.checkpoint_verified,
            source_compiler=source_parameters,
            query_parser=query_parameters,
            added_total=source_parameters + query_parameters,
            instantiated_total=(
                integrated_shohin + source_parameters + query_parameters
            ),
            hypothetical_complete_total=(
                protected_shohin + source_parameters + query_parameters
            ),
            global_limit=global_limit,
            hypothetical_headroom=(
                global_limit
                - protected_shohin
                - source_parameters
                - query_parameters
            ),
            integration_status="connected" if connected else "not_connected",
        )

    def parse_query(self, *args, **kwargs):
        del args, kwargs
        raise ConflictSystemError(
            "source-attached conflict system forbids same-process query parsing"
        )

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise ConflictSystemError(
            "source-attached conflict system requires a fresh-process runtime"
        )

    @staticmethod
    def execute_sealed(*args, **kwargs):
        del args, kwargs
        raise ConflictSystemError(
            "source-attached conflict system forbids same-process execution"
        )

    @torch.no_grad()
    def export_deployed_wire(
        self,
        compilation: ConflictCompilerOutput,
        *,
        batch_index: int = 0,
    ) -> bytes:
        sealed = self.seal(compilation)
        return sealed.machine.deployed_wire(
            sealed.keys,
            batch_index,
        )

    def compile_source(
        self,
        source: WitnessCompilerBatch,
        *,
        straight_through: bool = False,
        trunk_batch: ShohinTrunkBatch | None = None,
    ) -> ConflictCompilerOutput:
        del straight_through
        frozen_features = self._frozen_features(
            trunk_batch,
            byte_valid=source.pointer.byte_valid,
            label="source",
        )
        return self.source_compiler(
            source,
            straight_through=False,
            frozen_byte_features=frozen_features,
        )

    def seal(
        self,
        compilation: ConflictCompilerOutput,
    ) -> SealedFunctorBatch:
        if not isinstance(compilation, ConflictCompilerOutput):
            raise ConflictSystemError(
                "conflict system seal received the wrong compilation type"
            )
        sealed = self.source_compiler.seal(compilation)
        return SealedFunctorBatch(
            machine=sealed.machine,
            keys=sealed.keys,
        )


class ConflictQualificationLoss(EFCQualificationLoss):
    """Apply frozen binding labels to the second conflict-compiler pass."""

    def __init__(
        self,
        *,
        weights: QualificationLossWeights | None = None,
    ) -> None:
        super().__init__(weights=weights)

    @staticmethod
    def _final_witness(
        output: ConflictCompilerOutput,
    ) -> WitnessCompilerOutput:
        if not isinstance(output, ConflictCompilerOutput):
            raise ConflictSystemError(
                "conflict qualification output type differs"
            )
        return replace(
            output.witness,
            projection=output.revision.projection,
        )

    def forward(
        self,
        output: ConflictCompilerOutput,
        supervisor: QualificationSupervisorBatch,
        *,
        candidate_source_sha256: Sequence[str],
    ) -> QualificationLossOutput:
        return super().forward(
            self._final_witness(output),
            supervisor,
            candidate_source_sha256=candidate_source_sha256,
        )

    def exact_metrics(
        self,
        output: ConflictCompilerOutput,
        supervisor: QualificationSupervisorBatch,
        *,
        candidate_source_sha256: Sequence[str],
    ) -> QualificationExactMetrics:
        return super().exact_metrics(
            self._final_witness(output),
            supervisor,
            candidate_source_sha256=candidate_source_sha256,
        )


__all__ = [
    "ConflictQualificationLoss",
    "ConflictReentrantEFCSystem",
    "ConflictSystemError",
]
