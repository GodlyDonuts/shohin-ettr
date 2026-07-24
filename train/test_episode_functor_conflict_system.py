from __future__ import annotations

from hashlib import sha256

import pytest
import torch

from episode_functor_conflict_compiler import (
    ConflictCompilerOutput,
    ConflictProofCarryingCompiler,
)
from episode_functor_conflict_system import (
    ConflictQualificationLoss,
    ConflictReentrantEFCSystem,
    ConflictSystemError,
)
from episode_functor_detached_query_package import (
    detached_query_parser_state_sha256,
    export_detached_query_parser_package,
)
from episode_functor_query_parser import NeuralOpaqueQueryParser
from episode_functor_shohin_trunk import (
    FrozenShohinTrunk,
    ShohinTrunkParameterReceipt,
)
from episode_functor_witness_compiler import (
    collate_witness_sources,
    scan_witness_source,
)
from pipeline.episode_functor_identifiable_board import (
    GrammarFactors,
    encode_source,
    generate_machine,
    generate_pilot_rows,
    hide_one_cell_per_relation,
)
from pipeline.episode_functor_qualification_supervisor import (
    collate_qualification_supervision,
)


def _source() -> bytes:
    machine = generate_machine(
        seed="conflict-system-test-v1",
        split="mechanics",
        index=0,
        family="affine-f2-3",
    )
    evidence = hide_one_cell_per_relation(
        machine,
        seed="conflict-system-test-v1",
        split="mechanics",
        index=0,
    )
    return encode_source(evidence, GrammarFactors(0, 0, 0))


class _StubFrozenTrunk(FrozenShohinTrunk):
    def __init__(
        self,
        feature_width: int = 32,
        adapter_parameters: int = 0,
    ) -> None:
        torch.nn.Module.__init__(self)
        self._stub_feature_width = feature_width
        if adapter_parameters:
            self.adapter = torch.nn.Parameter(
                torch.zeros(adapter_parameters)
            )

    @property
    def feature_width(self) -> int:
        return self._stub_feature_width

    def parameter_receipt(self) -> ShohinTrunkParameterReceipt:
        adapter_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
        )
        return ShohinTrunkParameterReceipt(
            checkpoint_sha256="0" * 64,
            checkpoint_verified=False,
            parent_unique_parameters=125_081_664,
            adapter_unique_parameters=adapter_parameters,
            integrated_unique_parameters=(
                125_081_664 + adapter_parameters
            ),
            trainable_unique_parameters=adapter_parameters,
        )


def _preregister_parser(
    parser: NeuralOpaqueQueryParser,
    tmp_path,
    name: str,
):
    return export_detached_query_parser_package(
        parser,
        weights_path=tmp_path / f"{name}.safetensors",
        manifest_path=tmp_path / f"{name}.json",
    )


def test_system_compiles_twice_and_uses_second_pass_seal_type(
    tmp_path,
) -> None:
    torch.manual_seed(20260724)
    compiler = ConflictProofCarryingCompiler(
        external_feature_width=32,
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        controller_width=128,
        cycles=2,
    )
    parser = NeuralOpaqueQueryParser(
        width=48,
        layers=1,
        heads=3,
        feedforward=96,
        external_feature_width=0,
    )
    system = ConflictReentrantEFCSystem(
        source_compiler=compiler,
        query_parser=parser,
        query_parser_receipt=_preregister_parser(
            parser,
            tmp_path,
            "small",
        ),
        frozen_trunk=_StubFrozenTrunk(),
    )
    batch = collate_witness_sources(
        (scan_witness_source(_source()),)
    )
    frozen = torch.zeros(
        1,
        batch.pointer.byte_ids.shape[1],
        32,
    )
    output = compiler(
        batch,
        frozen_byte_features=frozen,
    )
    assert isinstance(output, ConflictCompilerOutput)
    assert output.witness.source_sha256 == output.first_witness.source_sha256
    with pytest.raises(
        ConflictSystemError,
        match="wrong compilation type",
    ):
        system.seal(output.witness)
    with pytest.raises(
        ConflictSystemError,
        match="forbids same-process query parsing",
    ):
        system.parse_query(None, None)
    with pytest.raises(
        ConflictSystemError,
        match="fresh-process runtime",
    ):
        system(None, None)
    with pytest.raises(
        ConflictSystemError,
        match="forbids same-process execution",
    ):
        system.execute_sealed(None, None)


def test_default_complete_system_receipt_remains_below_cap(
    tmp_path,
) -> None:
    with pytest.raises(
        ConflictSystemError,
        match="requires a frozen Shohin trunk",
    ):
        ConflictReentrantEFCSystem()
    detached_parser = NeuralOpaqueQueryParser(
        width=160,
        layers=2,
        heads=5,
        feedforward=640,
        external_feature_width=0,
    )
    package = export_detached_query_parser_package(
        detached_parser,
        weights_path=tmp_path / "parser.safetensors",
        manifest_path=tmp_path / "parser.json",
    )
    default_system = ConflictReentrantEFCSystem(
        query_parser=detached_parser,
        query_parser_receipt=package,
        frozen_trunk=_StubFrozenTrunk(1728),
    )
    assert isinstance(
        default_system.source_compiler,
        ConflictProofCarryingCompiler,
    )
    assert default_system.source_compiler.parameter_count() == 74_067_262
    assert not hasattr(default_system, "query_parser")
    assert default_system.detached_query_parameter_count == 748_033
    assert default_system.detached_query_architecture == (
        detached_parser.architecture_config()
    )
    assert default_system.detached_query_state_sha256 == (
        detached_query_parser_state_sha256(detached_parser)
    )
    assert (
        package.state_sha256
        == default_system.detached_query_state_sha256
    )
    assert (
        package.manifest_sha256
        == default_system.detached_query_manifest_sha256
    )
    assert (
        package.weights_sha256
        == default_system.detached_query_weights_sha256
    )
    assert default_system.added_parameter_count() == 74_815_295
    assert default_system.complete_parameter_count() == 199_896_959
    assert default_system.parameter_headroom() == 103_041
    assert sum(
        parameter.numel()
        for parameter in default_system.parameters()
    ) == 74_067_262
    receipt = default_system.parameter_receipt()
    assert receipt.query_parser == 748_033
    assert receipt.hypothetical_complete_total == 199_896_959
    assert receipt.hypothetical_headroom == 103_041


def test_constructor_rejects_oversized_detached_query_parser(
    tmp_path,
) -> None:
    parser = NeuralOpaqueQueryParser(
        width=256,
        layers=4,
        heads=8,
        feedforward=1024,
        external_feature_width=0,
    )
    with pytest.raises(
        ConflictSystemError,
        match="reaches or exceeds the 200M limit",
    ):
        ConflictReentrantEFCSystem(
            query_parser=parser,
            query_parser_receipt=_preregister_parser(
                parser,
                tmp_path,
                "oversized",
            ),
            frozen_trunk=_StubFrozenTrunk(1728),
        )


def test_constructor_rejects_query_parser_with_hidden_trunk_dependency(
    tmp_path,
) -> None:
    valid = NeuralOpaqueQueryParser(
        width=48,
        layers=1,
        heads=3,
        feedforward=96,
        external_feature_width=0,
    )
    with pytest.raises(
        ConflictSystemError,
        match="must be source-independent",
    ):
        ConflictReentrantEFCSystem(
            query_parser=NeuralOpaqueQueryParser(
                width=48,
                layers=1,
                heads=3,
                feedforward=96,
                external_feature_width=32,
            ),
            query_parser_receipt=_preregister_parser(
                valid,
                tmp_path,
                "valid-control",
            ),
            frozen_trunk=_StubFrozenTrunk(1728),
        )


def test_constructor_rejects_trunk_adapter_cap_bypass(tmp_path) -> None:
    parser = NeuralOpaqueQueryParser(
        width=48,
        layers=1,
        heads=3,
        feedforward=96,
        external_feature_width=0,
    )
    with pytest.raises(
        ConflictSystemError,
        match="exact adapter-free Shohin trunk",
    ):
        ConflictReentrantEFCSystem(
            query_parser=parser,
            query_parser_receipt=_preregister_parser(
                parser,
                tmp_path,
                "adapter-control",
            ),
            frozen_trunk=_StubFrozenTrunk(
                1728,
                adapter_parameters=122_082,
            ),
        )


def test_post_forward_supervision_trains_second_pass_and_binding_heads() -> None:
    torch.manual_seed(20260724)
    rows = generate_pilot_rows(
        seed="conflict-qualification-test-v1",
        counts={
            "train": 1,
            "mechanics": 1,
            "development": 1,
            "confirmation": 1,
        },
    )
    rows = tuple(row for row in rows if row.split == "train")
    batch = collate_witness_sources(
        tuple(scan_witness_source(row.source) for row in rows)
    )
    supervisor = collate_qualification_supervision(rows)
    hashes = tuple(sha256(row.source).hexdigest() for row in rows)
    compiler = ConflictProofCarryingCompiler(
        external_feature_width=32,
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        controller_width=128,
        cycles=2,
    )
    frozen = torch.zeros(
        len(rows),
        batch.pointer.byte_ids.shape[1],
        32,
    )
    output = compiler(
        batch,
        frozen_byte_features=frozen,
        straight_through=False,
    )
    objective = ConflictQualificationLoss()
    losses = objective(
        output,
        supervisor,
        candidate_source_sha256=hashes,
    )
    losses.total.backward()
    assert float(losses.total.detach()) > 0.0
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in compiler.parameters()
    )
    metrics = objective.exact_metrics(
        output,
        supervisor,
        candidate_source_sha256=hashes,
    )
    assert metrics.rows == len(rows)
    assert metrics.hidden_transition_cells == 3 * len(rows)
    assert metrics.hidden_observer_cells == 2 * len(rows)
