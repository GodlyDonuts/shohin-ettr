from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from episode_functor_detached_query_package import (
    build_detached_execution_authorization,
    export_detached_execution_authorization,
    export_detached_query_parser_package,
)
from episode_functor_machine import HardFunctorKeys, HardFunctorMachine
from episode_functor_query_parser import NeuralOpaqueQueryParser
from pipeline.episode_functor_detached_query_worker import (
    DetachedQueryWorkerError,
    MACHINE_HASH_OFFSET,
    _decode_wire,
    execute_detached_query,
)
from pipeline.episode_functor_identifiable_board import generate_machine


WORKER = Path(__file__).with_name(
    "episode_functor_detached_query_worker.py"
)


def _hard_machine(seed: str) -> tuple[HardFunctorMachine, HardFunctorKeys]:
    source = generate_machine(
        seed=seed,
        split="mechanics",
        index=0,
        family="affine-f2-3",
    )
    machine = HardFunctorMachine(
        state_active=torch.tensor(
            [[1] * 8 + [0] * 8],
            dtype=torch.uint8,
        ),
        action_active=torch.tensor(
            [[1] * 3 + [0] * 5],
            dtype=torch.uint8,
        ),
        observer_active=torch.tensor(
            [[1] * 2 + [0] * 6],
            dtype=torch.uint8,
        ),
        action_next=torch.tensor(
            [
                [
                    [*row, *([0] * 8)]
                    for row in source.transitions
                ]
                + [[0] * 16 for _ in range(5)]
            ],
            dtype=torch.uint8,
        ),
        observer_answer=torch.tensor(
            [
                [
                    [*row, *([0] * 8)]
                    for row in source.observations
                ]
                + [[0] * 16 for _ in range(6)]
            ],
            dtype=torch.uint8,
        ),
    )

    def key_rows(values: tuple[int, ...], maximum: int) -> torch.Tensor:
        payload = b"".join(
            value.to_bytes(8, "little")
            for value in values
        ) + b"\0" * (8 * (maximum - len(values)))
        return torch.tensor(tuple(payload), dtype=torch.uint8).reshape(
            1,
            maximum,
            8,
        )

    return machine, HardFunctorKeys(
        state_keys=key_rows(source.state_keys, 16),
        action_keys=key_rows(source.action_keys, 8),
        observer_keys=key_rows(source.observer_keys, 8),
    )


def _token(value: torch.Tensor) -> bytes:
    integer = int.from_bytes(bytes(value), "little")
    return f"h{integer:016x}".encode("ascii")


def test_fresh_neural_worker_uses_only_bound_parser_wire_and_late_query(
    tmp_path: Path,
) -> None:
    torch.manual_seed(20260724)
    parser = NeuralOpaqueQueryParser(
        width=32,
        layers=1,
        heads=4,
        feedforward=64,
        max_steps=8,
        external_feature_width=0,
    )
    weights = tmp_path / "parser.safetensors"
    manifest = tmp_path / "parser.json"
    receipt = export_detached_query_parser_package(
        parser,
        weights_path=weights,
        manifest_path=manifest,
    )
    machine, keys = _hard_machine("detached-neural-worker-v1")
    wire = machine.deployed_wire(keys, 0)
    authorization = build_detached_execution_authorization(
        machine_sha256=sha256(wire).hexdigest(),
        parser_receipt=receipt,
        source_compiler_parameter_count=74_067_262,
        source_compiler_state_sha256="1" * 64,
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_sha256 = export_detached_execution_authorization(
        authorization,
        path=authorization_path,
    )
    query = b" ".join(
        (
            b"START",
            _token(keys.state_keys[0, 3]),
            b"THEN",
            _token(keys.action_keys[0, 2]),
            b"THEN",
            _token(keys.action_keys[0, 0]),
            b"WATCH",
            _token(keys.observer_keys[0, 1]),
        )
    )
    expected = execute_detached_query(
        machine_payload=wire,
        query_payload=query,
        weights_path=weights,
        manifest_path=manifest,
        authorization_path=authorization_path,
        expected_authorization_sha256=authorization_sha256,
    )

    source = tmp_path / "source.txt"
    machine_path = tmp_path / "machine.bin"
    query_path = tmp_path / "query.bin"
    output_path = tmp_path / "output.json"
    source.write_text("must be deleted before late-query generation")
    machine_path.write_bytes(wire)
    query_path.write_bytes(query)
    source.unlink()
    machine.action_next.zero_()
    machine.observer_answer.zero_()
    keys.state_keys.zero_()
    keys.action_keys.zero_()
    keys.observer_keys.zero_()
    subprocess.run(
        (
            sys.executable,
            "-I",
            str(WORKER),
            "--machine",
            str(machine_path),
            "--query",
            str(query_path),
            "--parser-weights",
            str(weights),
            "--parser-manifest",
            str(manifest),
            "--authorization",
            str(authorization_path),
            "--expected-authorization-sha256",
            authorization_sha256,
            "--output",
            str(output_path),
        ),
        cwd=tmp_path,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    actual = json.loads(output_path.read_text())
    assert not source.exists()
    assert actual == expected
    assert (
        actual["parser_receipt"]["manifest_sha256"]
        == receipt.manifest_sha256
    )
    worker_source = WORKER.read_text()
    assert "import episode_functor_conflict_compiler" not in worker_source
    assert "import episode_functor_witness_compiler" not in worker_source


@pytest.mark.parametrize("offset", (20, 30, 56, 1_472))
def test_neural_worker_rejects_noncanonical_reserved_wire(
    offset: int,
) -> None:
    machine, keys = _hard_machine("detached-neural-worker-reserved-v1")
    payload = bytearray(machine.deployed_wire(keys, 0))
    payload[offset] = 1
    payload[MACHINE_HASH_OFFSET:] = sha256(
        payload[:MACHINE_HASH_OFFSET]
    ).digest()
    with pytest.raises(
        DetachedQueryWorkerError,
        match="noncanonical",
    ):
        _decode_wire(bytes(payload))
