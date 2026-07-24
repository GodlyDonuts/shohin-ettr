from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import stat

import pytest
import torch

from episode_functor_detached_query_package import (
    DetachedQueryPackageError,
    detached_query_parser_state_sha256,
    export_detached_query_parser_package,
    load_detached_query_parser_package,
)
from episode_functor_query_parser import NeuralOpaqueQueryParser


def _parser(*, external_feature_width: int = 0) -> NeuralOpaqueQueryParser:
    torch.manual_seed(20260724)
    return NeuralOpaqueQueryParser(
        width=32,
        layers=1,
        heads=4,
        feedforward=64,
        max_steps=8,
        external_feature_width=external_feature_width,
    )


def test_package_binds_exact_config_weights_and_state_schema(
    tmp_path: Path,
) -> None:
    parser = _parser()
    weights = tmp_path / "parser.safetensors"
    manifest = tmp_path / "parser.json"
    receipt = export_detached_query_parser_package(
        parser,
        weights_path=weights,
        manifest_path=manifest,
    )
    loaded, replay = load_detached_query_parser_package(
        weights_path=weights,
        manifest_path=manifest,
        expected_manifest_sha256=receipt.manifest_sha256,
    )
    assert replay == receipt
    assert replay.weights_sha256 == sha256(weights.read_bytes()).hexdigest()
    assert replay.state_sha256 == detached_query_parser_state_sha256(parser)
    assert loaded.architecture_config() == parser.architecture_config()
    assert loaded.parameter_count() == parser.parameter_count()
    assert not loaded.training
    assert stat.S_IMODE(weights.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600
    for (expected_name, expected), (actual_name, actual) in zip(
        parser.state_dict().items(),
        loaded.state_dict().items(),
        strict=True,
    ):
        assert actual_name == expected_name
        assert torch.equal(actual, expected)


def test_package_fails_closed_on_weight_or_manifest_mutation(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "parser.safetensors"
    manifest = tmp_path / "parser.json"
    receipt = export_detached_query_parser_package(
        _parser(),
        weights_path=weights,
        manifest_path=manifest,
    )
    changed = bytearray(weights.read_bytes())
    changed[-1] ^= 1
    weights.write_bytes(changed)
    with pytest.raises(
        DetachedQueryPackageError,
        match="weights differ",
    ):
        load_detached_query_parser_package(
            weights_path=weights,
            manifest_path=manifest,
            expected_manifest_sha256=receipt.manifest_sha256,
        )

    weights.unlink()
    manifest.unlink()
    receipt = export_detached_query_parser_package(
        _parser(),
        weights_path=weights,
        manifest_path=manifest,
    )
    manifest.write_bytes(manifest.read_bytes().replace(b'"width":32', b'"width":34'))
    with pytest.raises(
        DetachedQueryPackageError,
        match="manifest hash differs",
    ):
        load_detached_query_parser_package(
            weights_path=weights,
            manifest_path=manifest,
            expected_manifest_sha256=receipt.manifest_sha256,
        )


def test_package_rejects_hidden_frozen_feature_dependency(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        DetachedQueryPackageError,
        match="external feature dependency",
    ):
        export_detached_query_parser_package(
            _parser(external_feature_width=32),
            weights_path=tmp_path / "parser.safetensors",
            manifest_path=tmp_path / "parser.json",
        )


def test_package_outputs_are_no_clobber(tmp_path: Path) -> None:
    weights = tmp_path / "parser.safetensors"
    manifest = tmp_path / "parser.json"
    export_detached_query_parser_package(
        _parser(),
        weights_path=weights,
        manifest_path=manifest,
    )
    with pytest.raises(
        DetachedQueryPackageError,
        match="new absolute siblings",
    ):
        export_detached_query_parser_package(
            _parser(),
            weights_path=weights,
            manifest_path=manifest,
        )
