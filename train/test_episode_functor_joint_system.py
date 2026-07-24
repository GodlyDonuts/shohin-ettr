from __future__ import annotations

from hashlib import sha256
import inspect

import torch
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from episode_functor_joint_system import (
    JointEquilibriumEFCSystem,
    JointSystemError,
    _read_bound_tokenizer,
)
from episode_functor_joint_compiler import JointProofCarryingCompiler
from episode_functor_shohin_trunk import (
    FrozenShohinTrunk,
    ShohinTrunkBatch,
    ShohinTrunkParameterReceipt,
)
from pipeline.episode_functor_qualification_boundary import (
    tokenizer_runtime_sha256,
)


class _StubFrozenTrunk(FrozenShohinTrunk):
    def __init__(self, feature_width: int = 1728) -> None:
        torch.nn.Module.__init__(self)
        self._stub_feature_width = feature_width

    @property
    def feature_width(self) -> int:
        return self._stub_feature_width

    def parameter_receipt(self) -> ShohinTrunkParameterReceipt:
        return ShohinTrunkParameterReceipt(
            checkpoint_sha256="0" * 64,
            checkpoint_verified=False,
            parent_unique_parameters=125_081_664,
            adapter_unique_parameters=0,
            integrated_unique_parameters=125_081_664,
            trainable_unique_parameters=0,
        )


class _ByteEncoding:
    def __init__(self, payload: bytes) -> None:
        self.ids = tuple(int(value) for value in payload)
        self.offsets = tuple(
            (index, index + 1) for index in range(len(payload))
        )


class _ByteTokenizer:
    def encode(self, value: str) -> _ByteEncoding:
        return _ByteEncoding(value.encode("ascii"))


def test_joint_system_rejects_trunk_subclasses() -> None:
    with pytest.raises(
        JointSystemError,
        match="exact frozen Shohin trunk type",
    ):
        JointEquilibriumEFCSystem(
            frozen_trunk=_StubFrozenTrunk(),
        )


def test_joint_system_rejects_exact_type_self_attestation() -> None:
    forged = FrozenShohinTrunk.__new__(FrozenShohinTrunk)
    torch.nn.Module.__init__(forged)
    forged.parameter_receipt = lambda: ShohinTrunkParameterReceipt(
        checkpoint_sha256="0" * 64,
        checkpoint_verified=True,
        parent_unique_parameters=125_081_664,
        adapter_unique_parameters=0,
        integrated_unique_parameters=125_081_664,
        trainable_unique_parameters=0,
    )
    with pytest.raises(
        JointSystemError,
        match="custody cannot be verified",
    ):
        JointEquilibriumEFCSystem(frozen_trunk=forged)


def test_joint_system_rejects_raw_wire_authorization() -> None:
    system = JointEquilibriumEFCSystem.__new__(
        JointEquilibriumEFCSystem
    )
    with pytest.raises(
        JointSystemError,
        match="forbids raw-wire authorization",
    ):
        system.authorize_deployed_wire(bytes(1_536))


def test_joint_system_rejects_nonanonymous_trunk_payload() -> None:
    system = JointEquilibriumEFCSystem.__new__(
        JointEquilibriumEFCSystem
    )
    trunk_batch = ShohinTrunkBatch(
        payloads=(b"raw opaque source",),
        token_ids=torch.zeros((1, 1), dtype=torch.long),
        token_valid=torch.ones((1, 1), dtype=torch.bool),
        token_byte_bounds=torch.zeros((1, 1, 2), dtype=torch.int32),
    )
    with pytest.raises(
        JointSystemError,
        match="anonymous source view",
    ):
        system._frozen_features(
            trunk_batch,
            byte_valid=torch.ones((1, 2), dtype=torch.bool),
            label="source",
            expected_payloads=(b"d1",),
        )


def test_joint_system_owns_anonymous_tokenization() -> None:
    assert "trunk_batch" not in inspect.signature(
        JointEquilibriumEFCSystem.compile_source
    ).parameters
    system = JointEquilibriumEFCSystem.__new__(
        JointEquilibriumEFCSystem
    )
    system._source_tokenizer = _ByteTokenizer()
    batch = system._tokenize_anonymous_payloads(
        (b"d1xx",),
        device=torch.device("cpu"),
    )
    assert batch.payloads == (b"d1xx",)
    assert batch.token_ids.tolist() == [[100, 49, 120, 120]]
    assert batch.token_valid.tolist() == [[True, True, True, True]]
    assert batch.token_byte_bounds.tolist() == [
        [[0, 1], [1, 2], [2, 3], [3, 4]]
    ]


def test_joint_system_loads_exact_tokenizer_artifact(tmp_path) -> None:
    tokenizer = Tokenizer(
        WordLevel(vocab={"[UNK]": 0, "d1": 1}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    encoded = tokenizer.to_str().encode("utf-8")
    path.write_bytes(encoded)
    runtime_sha256 = tokenizer_runtime_sha256(tokenizer)
    loaded = _read_bound_tokenizer(
        path,
        expected_artifact_sha256=sha256(encoded).hexdigest(),
        expected_runtime_sha256=runtime_sha256,
    )
    assert tokenizer_runtime_sha256(loaded) == runtime_sha256
    with pytest.raises(
        JointSystemError,
        match="artifact hash differs",
    ):
        _read_bound_tokenizer(
            path,
            expected_artifact_sha256="0" * 64,
            expected_runtime_sha256=runtime_sha256,
        )


def test_joint_system_recomputes_live_compiler_parameter_cap() -> None:
    system = JointEquilibriumEFCSystem.__new__(
        JointEquilibriumEFCSystem
    )
    torch.nn.Module.__init__(system)
    compiler = JointProofCarryingCompiler(
        external_feature_width=0,
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        assignment_width=48,
        assignment_context_width=96,
        machine_width=64,
        machine_context_width=128,
        cycles=2,
        sinkhorn_iterations=16,
    )
    system.source_compiler = compiler
    system._source_compiler_parameters = 0
    system._detached_query_parameters = 748_033
    live = sum(parameter.numel() for parameter in compiler.parameters())
    assert system._live_source_compiler_parameter_count() == live
    fill = 200_000_000 - 125_081_664 - 748_033 - live
    compiler.register_parameter(
        "post_construction_parameter",
        torch.nn.Parameter(torch.empty(fill, device="meta")),
    )
    with pytest.raises(
        JointSystemError,
        match="reaches or exceeds",
    ):
        system._live_source_compiler_parameter_count()
