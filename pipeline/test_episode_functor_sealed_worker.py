from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import torch

from episode_functor_machine import HardFunctorKeys, HardFunctorMachine
from pipeline.episode_functor_identifiable_board import generate_machine


WORKER = Path(__file__).with_name("episode_functor_sealed_worker.py")


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

    keys = HardFunctorKeys(
        state_keys=key_rows(source.state_keys, 16),
        action_keys=key_rows(source.action_keys, 8),
        observer_keys=key_rows(source.observer_keys, 8),
    )
    return machine, keys


def test_fresh_worker_executes_only_wire_after_source_deletion(
    tmp_path: Path,
) -> None:
    machine, keys = _hard_machine("sealed-worker-test-v1")
    wire = machine.deployed_wire(keys, 0)
    start = int.from_bytes(bytes(keys.state_keys[0, 3]), "little")
    actions = tuple(
        int.from_bytes(bytes(keys.action_keys[0, index]), "little")
        for index in (2, 0, 2)
    )
    observer = int.from_bytes(
        bytes(keys.observer_keys[0, 1]),
        "little",
    )
    expected_state = 3
    for action in (2, 0, 2):
        expected_state = int(machine.action_next[0, action, expected_state])
    expected_answer = int(
        machine.observer_answer[0, 1, expected_state]
    )
    query = {
        "action_keys": actions,
        "observer_key": observer,
        "start_key": start,
    }
    query_payload = (
        json.dumps(query, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    source = tmp_path / "source.json"
    machine_path = tmp_path / "machine.bin"
    query_path = tmp_path / "query.json"
    output_path = tmp_path / "output.json"
    source.write_text(json.dumps(asdict(generate_machine(
        seed="sealed-worker-test-v1",
        split="mechanics",
        index=0,
        family="affine-f2-3",
    ))))
    machine_path.write_bytes(wire)
    query_path.write_bytes(query_payload)
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
            "--output",
            str(output_path),
        ),
        cwd=tmp_path,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    result = json.loads(output_path.read_text())
    assert not source.exists()
    assert result["answer"] == expected_answer
    assert result["final_state_slot"] == expected_state
    assert result["final_state_key"] == int.from_bytes(
        wire[64 + expected_state * 8 : 72 + expected_state * 8],
        "little",
    )
    worker_source = WORKER.read_text()
    assert "torch" not in worker_source
    assert "conflict_compiler" not in worker_source
    assert "witness" not in worker_source
