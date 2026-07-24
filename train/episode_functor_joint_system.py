"""Source-deleted EFC system backed by the JASEC source compiler."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import stat
from typing import Sequence

import torch

from pipeline.episode_functor_qualification_batch import (
    QualificationSupervisorBatch,
)
from pipeline.episode_functor_qualification_boundary import (
    tokenizer_runtime_sha256,
)
from episode_functor_conflict_system import (
    ConflictReentrantEFCSystem,
    MAXIMUM_SYSTEM_PARAMETERS,
    PROTECTED_SHOHIN_PARAMETERS,
)
from episode_functor_detached_query_package import (
    DetachedExecutionAuthorization,
    DetachedQueryParserReceipt,
    PACKAGE_SCHEMA,
    build_detached_execution_authorization,
    detached_query_parser_state_sha256,
    load_detached_query_parser_package,
)
from episode_functor_joint_compiler import (
    JointCompilerOutput,
    JointProofCarryingCompiler,
    SealedJointMachine,
)
from episode_functor_learned_system import PROTECTED_SHOHIN_SHA256
from episode_functor_learned_system import (
    EFCParameterReceipt,
    GLOBAL_PARAMETER_LIMIT,
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
    ShohinTrunkParameterReceipt,
)
from episode_functor_witness_compiler import (
    WitnessCompilerBatch,
    WitnessCompilerOutput,
    canonicalize_witness_batch,
)


class JointSystemError(ValueError):
    """JASEC system compilation, supervision, or seal failed closed."""


def _read_bound_tokenizer(
    path: Path,
    *,
    expected_artifact_sha256: str,
    expected_runtime_sha256: str,
):
    from tokenizers import Tokenizer

    if (
        not isinstance(expected_artifact_sha256, str)
        or len(expected_artifact_sha256) != 64
        or not isinstance(expected_runtime_sha256, str)
        or len(expected_runtime_sha256) != 64
    ):
        raise JointSystemError("joint system tokenizer receipt differs")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise JointSystemError(
            "joint system tokenizer artifact cannot be opened"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size < 1
            or metadata.st_size > 128 * 1024 * 1024
        ):
            raise JointSystemError(
                "joint system tokenizer artifact differs"
            )
        raw = bytearray()
        while len(raw) <= 128 * 1024 * 1024:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            raw.extend(chunk)
    finally:
        os.close(descriptor)
    encoded = bytes(raw)
    if (
        len(encoded) != metadata.st_size
        or sha256(encoded).hexdigest() != expected_artifact_sha256
    ):
        raise JointSystemError(
            "joint system tokenizer artifact hash differs"
        )
    try:
        tokenizer = Tokenizer.from_str(encoded.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise JointSystemError(
            "joint system tokenizer artifact is malformed"
        ) from error
    if tokenizer_runtime_sha256(tokenizer) != expected_runtime_sha256:
        raise JointSystemError(
            "joint system tokenizer runtime hash differs"
        )
    return tokenizer


def _verified_trunk_receipt(
    trunk: FrozenShohinTrunk | None,
) -> ShohinTrunkParameterReceipt:
    if type(trunk) is not FrozenShohinTrunk:
        raise JointSystemError(
            "joint system requires the exact frozen Shohin trunk type"
        )
    try:
        receipt = FrozenShohinTrunk.parameter_receipt(trunk)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise JointSystemError(
            "joint system trunk custody cannot be verified"
        ) from error
    if type(receipt) is not ShohinTrunkParameterReceipt:
        raise JointSystemError(
            "joint system trunk receipt type differs"
        )
    return receipt


class JointEquilibriumEFCSystem(ConflictReentrantEFCSystem):
    """Frozen Shohin plus JASEC and a detached late-query parser."""

    source_compiler: JointProofCarryingCompiler

    def __init__(
        self,
        *,
        source_compiler: JointProofCarryingCompiler | None = None,
        query_parser_weights_path: Path | None = None,
        query_parser_manifest_path: Path | None = None,
        expected_query_parser_manifest_sha256: str | None = None,
        tokenizer_path: Path | None = None,
        expected_tokenizer_artifact_sha256: str | None = None,
        expected_tokenizer_runtime_sha256: str | None = None,
        frozen_trunk: FrozenShohinTrunk | None = None,
    ) -> None:
        trunk_receipt = _verified_trunk_receipt(frozen_trunk)
        if (
            not trunk_receipt.checkpoint_verified
            or trunk_receipt.parent_unique_parameters
            != PROTECTED_SHOHIN_PARAMETERS
            or trunk_receipt.adapter_unique_parameters != 0
            or trunk_receipt.integrated_unique_parameters
            != PROTECTED_SHOHIN_PARAMETERS
            or trunk_receipt.trainable_unique_parameters != 0
        ):
            raise JointSystemError(
                "joint system requires verified adapter-free Shohin custody"
            )
        if (
            query_parser_weights_path is None
            or query_parser_manifest_path is None
            or expected_query_parser_manifest_sha256 is None
        ):
            raise JointSystemError(
                "joint system requires an exact detached parser package"
            )
        if (
            tokenizer_path is None
            or expected_tokenizer_artifact_sha256 is None
            or expected_tokenizer_runtime_sha256 is None
        ):
            raise JointSystemError(
                "joint system requires an exact tokenizer package"
            )
        source_tokenizer = _read_bound_tokenizer(
            Path(tokenizer_path),
            expected_artifact_sha256=(
                expected_tokenizer_artifact_sha256
            ),
            expected_runtime_sha256=expected_tokenizer_runtime_sha256,
        )
        parser, query_parser_receipt = (
            load_detached_query_parser_package(
                weights_path=Path(query_parser_weights_path),
                manifest_path=Path(query_parser_manifest_path),
                expected_manifest_sha256=(
                    expected_query_parser_manifest_sha256
                ),
            )
        )
        feature_width_property = FrozenShohinTrunk.feature_width
        if feature_width_property.fget is None:
            raise JointSystemError(
                "joint system trunk feature geometry differs"
            )
        frozen_feature_width = int(
            feature_width_property.fget(frozen_trunk)
        )
        compiler = (
            JointProofCarryingCompiler(
                external_feature_width=frozen_feature_width,
            )
            if source_compiler is None
            else source_compiler
        )
        if type(compiler) is not JointProofCarryingCompiler:
            raise JointSystemError(
                "joint system requires the JASEC source compiler"
            )
        if compiler.external_feature_width != frozen_feature_width:
            raise JointSystemError(
                "joint compiler frozen-feature width differs"
            )
        source_compiler_parameters = sum(
            parameter.numel() for parameter in compiler.parameters()
        )
        if type(parser) is not NeuralOpaqueQueryParser:
            raise JointSystemError(
                "detached query parser must have the exact approved type"
            )
        if parser.external_feature_width != 0:
            raise JointSystemError(
                "detached query parser must be source-independent"
            )
        detached_query_parameters = sum(
            parameter.numel() for parameter in parser.parameters()
        )
        parser_architecture = parser.architecture_config()
        parser_state_sha256 = detached_query_parser_state_sha256(parser)
        if (
            type(query_parser_receipt) is not DetachedQueryParserReceipt
            or query_parser_receipt.schema != PACKAGE_SCHEMA
            or query_parser_receipt.parameter_count
            != detached_query_parameters
            or query_parser_receipt.architecture != parser_architecture
            or query_parser_receipt.state_sha256 != parser_state_sha256
        ):
            raise JointSystemError(
                "detached parser differs from its preregistered package"
            )
        torch.nn.Module.__init__(self)
        self.source_compiler = compiler
        self.frozen_trunk = frozen_trunk
        complete_parameters = (
            trunk_receipt.integrated_unique_parameters
            + source_compiler_parameters
            + detached_query_parameters
        )
        if complete_parameters >= MAXIMUM_SYSTEM_PARAMETERS:
            raise JointSystemError(
                "joint system reaches or exceeds the 200M limit"
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
        self._source_tokenizer = source_tokenizer
        self._tokenizer_artifact_sha256 = (
            expected_tokenizer_artifact_sha256
        )
        self._tokenizer_runtime_sha256 = (
            expected_tokenizer_runtime_sha256
        )

    def _live_source_compiler_parameter_count(self) -> int:
        if type(self.source_compiler) is not JointProofCarryingCompiler:
            raise JointSystemError(
                "joint system source compiler type differs"
            )
        count = sum(
            parameter.numel()
            for parameter in self.source_compiler.parameters()
        )
        if (
            PROTECTED_SHOHIN_PARAMETERS
            + count
            + self._detached_query_parameters
            >= MAXIMUM_SYSTEM_PARAMETERS
        ):
            raise JointSystemError(
                "joint system reaches or exceeds the 200M limit"
            )
        return count

    def added_parameter_count(self) -> int:
        return (
            self._live_source_compiler_parameter_count()
            + self._detached_query_parameters
        )

    def complete_parameter_count(self) -> int:
        return PROTECTED_SHOHIN_PARAMETERS + self.added_parameter_count()

    def parameter_headroom(self) -> int:
        return MAXIMUM_SYSTEM_PARAMETERS - self.complete_parameter_count()

    def parameter_receipt(
        self,
        *,
        protected_shohin: int = PROTECTED_SHOHIN_PARAMETERS,
        protected_checkpoint_sha256: str = PROTECTED_SHOHIN_SHA256,
        global_limit: int = GLOBAL_PARAMETER_LIMIT,
    ) -> EFCParameterReceipt:
        trunk_receipt = _verified_trunk_receipt(self.frozen_trunk)
        source_parameters = self._live_source_compiler_parameter_count()
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

    def _tokenize_anonymous_payloads(
        self,
        payloads: tuple[bytes, ...],
        *,
        device: torch.device,
    ) -> ShohinTrunkBatch:
        encoded: list[
            tuple[tuple[int, ...], tuple[tuple[int, int], ...]]
        ] = []
        for payload in payloads:
            try:
                result = self._source_tokenizer.encode(
                    payload.decode("ascii")
                )
                ids = tuple(int(value) for value in result.ids)
                offsets = tuple(
                    (int(start), int(end))
                    for start, end in result.offsets
                )
            except (
                AttributeError,
                TypeError,
                UnicodeDecodeError,
                ValueError,
            ) as error:
                raise JointSystemError(
                    "joint system anonymous tokenization failed"
                ) from error
            coverage = [0] * len(payload)
            for start, end in offsets:
                if not 0 <= start < end <= len(payload):
                    raise JointSystemError(
                        "joint tokenizer offset leaves anonymous source"
                    )
                for index in range(start, end):
                    coverage[index] += 1
            if (
                not ids
                or len(ids) != len(offsets)
                or any(value < 0 for value in ids)
                or any(value != 1 for value in coverage)
            ):
                raise JointSystemError(
                    "joint tokenizer does not partition anonymous source"
                )
            encoded.append((ids, offsets))
        maximum_tokens = max(len(ids) for ids, _ in encoded)
        token_ids = torch.zeros(
            (len(payloads), maximum_tokens),
            dtype=torch.long,
            device=device,
        )
        token_valid = torch.zeros(
            (len(payloads), maximum_tokens),
            dtype=torch.bool,
            device=device,
        )
        token_bounds = torch.zeros(
            (len(payloads), maximum_tokens, 2),
            dtype=torch.int32,
            device=device,
        )
        for row, (ids, offsets) in enumerate(encoded):
            count = len(ids)
            token_ids[row, :count] = torch.tensor(
                ids,
                dtype=torch.long,
                device=device,
            )
            token_valid[row, :count] = True
            token_bounds[row, :count] = torch.tensor(
                offsets,
                dtype=torch.int32,
                device=device,
            )
        return ShohinTrunkBatch(
            payloads=payloads,
            token_ids=token_ids,
            token_valid=token_valid,
            token_byte_bounds=token_bounds,
        )

    def _frozen_features(
        self,
        trunk_batch: ShohinTrunkBatch | None,
        *,
        byte_valid: torch.Tensor,
        label: str,
        expected_payloads: tuple[bytes, ...],
    ) -> torch.Tensor:
        if trunk_batch is None:
            raise JointSystemError(
                f"connected system is missing {label} trunk input"
            )
        if trunk_batch.payloads != expected_payloads:
            raise JointSystemError(
                f"{label} trunk did not use the anonymous source view"
            )
        features = FrozenShohinTrunk.encode_batch(
            self.frozen_trunk,
            trunk_batch,
        )
        flattened = FrozenShohinTrunk.flatten_byte_features(
            self.frozen_trunk,
            features,
        )
        if (
            flattened.shape[:2] != byte_valid.shape
            or features.byte_valid.device != byte_valid.device
            or not bool(torch.equal(features.byte_valid, byte_valid))
        ):
            raise JointSystemError(
                f"{label} trunk byte alignment differs from parser input"
            )
        return flattened

    def compile_source(
        self,
        source: WitnessCompilerBatch,
        *,
        straight_through: bool = False,
        mode: str = "causal",
    ) -> JointCompilerOutput:
        if straight_through:
            raise JointSystemError(
                "joint system forbids solver-backed training assignment"
            )
        model_source = canonicalize_witness_batch(source)
        payloads = tuple(
            bytes(
                model_source.pointer.byte_ids[
                    row,
                    : int(model_source.pointer.byte_valid[row].sum()),
                ]
                .detach()
                .cpu()
                .tolist()
            )
            for row in range(model_source.batch_size)
        )
        anonymous_trunk_batch = self._tokenize_anonymous_payloads(
            payloads,
            device=model_source.pointer.byte_ids.device,
        )
        frozen_features = self._frozen_features(
            anonymous_trunk_batch,
            byte_valid=model_source.pointer.byte_valid,
            label="source",
            expected_payloads=payloads,
        )
        return self.source_compiler(
            source,
            straight_through=False,
            frozen_byte_features=frozen_features,
            mode=mode,
        )

    def seal(
        self,
        compilation: JointCompilerOutput,
    ) -> SealedJointMachine:
        if type(compilation) is not JointCompilerOutput:
            raise JointSystemError(
                "joint system seal received the wrong compilation type"
            )
        return self.source_compiler.seal(compilation)

    def authorize_deployed_wire(
        self,
        wire: bytes,
    ) -> DetachedExecutionAuthorization:
        del wire
        raise JointSystemError(
            "joint system forbids raw-wire authorization; "
            "authorize a provenanced seal"
        )

    def authorize_sealed(
        self,
        sealed: SealedJointMachine,
        *,
        row: int,
    ) -> DetachedExecutionAuthorization:
        if type(self.frozen_trunk) is not FrozenShohinTrunk:
            raise JointSystemError(
                "deployment authorization requires the exact trunk type"
            )
        trunk_receipt = _verified_trunk_receipt(self.frozen_trunk)
        if (
            not trunk_receipt.checkpoint_verified
            or trunk_receipt.checkpoint_sha256
            != PROTECTED_SHOHIN_SHA256
            or trunk_receipt.parent_unique_parameters
            != PROTECTED_SHOHIN_PARAMETERS
            or trunk_receipt.adapter_unique_parameters != 0
            or trunk_receipt.integrated_unique_parameters
            != PROTECTED_SHOHIN_PARAMETERS
            or trunk_receipt.trainable_unique_parameters != 0
        ):
            raise JointSystemError(
                "deployment authorization requires verified Shohin custody"
            )
        wires = self.source_compiler.verify_sealed(sealed)
        if type(row) is not int or not 0 <= row < len(wires):
            raise JointSystemError(
                "deployment authorization row differs"
            )
        compiler_state_sha256 = self.source_compiler.deployment_sha256()
        source_compiler_parameters = (
            self._live_source_compiler_parameter_count()
        )
        return build_detached_execution_authorization(
            machine_sha256=sealed.wire_sha256[row],
            parser_receipt=self._detached_query_receipt,
            source_compiler_parameter_count=source_compiler_parameters,
            source_compiler_state_sha256=compiler_state_sha256,
        )


class JointQualificationLoss(EFCQualificationLoss):
    """Apply frozen binding and machine labels after the JASEC forward."""

    def __init__(
        self,
        *,
        weights: QualificationLossWeights | None = None,
    ) -> None:
        super().__init__(weights=weights)

    @staticmethod
    def _final_witness(
        output: JointCompilerOutput,
    ) -> WitnessCompilerOutput:
        if not isinstance(output, JointCompilerOutput):
            raise JointSystemError(
                "joint qualification output type differs"
            )
        return replace(
            output.witness,
            projection=output.witness.projection,
        )

    def forward(
        self,
        output: JointCompilerOutput,
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
        output: JointCompilerOutput,
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
    "JointEquilibriumEFCSystem",
    "JointQualificationLoss",
    "JointSystemError",
]
