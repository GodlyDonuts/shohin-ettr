"""Fresh-process neural query parsing and execution from a sealed EFC wire."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import struct
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from episode_functor_detached_query_package import (  # noqa: E402
    load_detached_execution_authorization,
    load_detached_query_parser_package,
    receipt_payload,
)
from episode_functor_machine import (  # noqa: E402
    HardFunctorKeys,
    HardFunctorMachine,
    MAX_ACTIONS,
    MAX_OBSERVERS,
    MAX_STATES,
    execute_hard,
)
from episode_functor_query_parser import (  # noqa: E402
    collate_queries,
    scan_query,
)


MACHINE_BYTES = 1_536
MACHINE_HASH_OFFSET = 1_504


class DetachedQueryWorkerError(ValueError):
    """The machine wire, parser package, query, or output failed closed."""


def _assert_detached_import_closure() -> None:
    forbidden = {
        "episode_functor_conflict_compiler",
        "episode_functor_constrained_transport",
        "episode_functor_pointer_compiler",
        "episode_functor_witness_compiler",
    }
    loaded = forbidden.intersection(sys.modules)
    if loaded:
        raise DetachedQueryWorkerError(
            "detached worker imported source-side modules: "
            + ",".join(sorted(loaded))
        )


def _u16(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<H", payload, offset)[0]


def _u32(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<I", payload, offset)[0]


def _tensor(payload: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    expected = 1
    for dimension in shape:
        expected *= dimension
    if len(payload) != expected:
        raise DetachedQueryWorkerError("wire tensor byte count differs")
    return torch.tensor(tuple(payload), dtype=torch.uint8).reshape(shape)


def _decode_wire(
    payload: bytes,
) -> tuple[HardFunctorMachine, HardFunctorKeys]:
    if (
        len(payload) != MACHINE_BYTES
        or payload[:8] != b"EFCMACH\0"
        or _u32(payload, 8) != 1
        or _u32(payload, 12) != 64
        or _u32(payload, 16) != MACHINE_BYTES
        or payload[MACHINE_HASH_OFFSET:]
        != sha256(payload[:MACHINE_HASH_OFFSET]).digest()
    ):
        raise DetachedQueryWorkerError("sealed machine wire is invalid")
    state_count = _u16(payload, 24)
    action_count = _u16(payload, 26)
    observer_count = _u16(payload, 28)
    if (
        state_count != 8
        or action_count != 3
        or observer_count != 2
    ):
        raise DetachedQueryWorkerError("sealed machine geometry is invalid")
    state_active = torch.zeros((1, MAX_STATES), dtype=torch.uint8)
    action_active = torch.zeros((1, MAX_ACTIONS), dtype=torch.uint8)
    observer_active = torch.zeros((1, MAX_OBSERVERS), dtype=torch.uint8)
    state_active[:, :state_count] = 1
    action_active[:, :action_count] = 1
    observer_active[:, :observer_count] = 1
    if (
        struct.unpack_from("<Q", payload, 32)[0]
        != (1 << state_count) - 1
        or struct.unpack_from("<Q", payload, 40)[0]
        != (1 << action_count) - 1
        or struct.unpack_from("<Q", payload, 48)[0]
        != (1 << observer_count) - 1
    ):
        raise DetachedQueryWorkerError("sealed active masks differ")
    machine = HardFunctorMachine(
        state_active=state_active,
        action_active=action_active,
        observer_active=observer_active,
        action_next=_tensor(
            payload[320:448],
            (1, MAX_ACTIONS, MAX_STATES),
        ),
        observer_answer=torch.tensor(
            [
                struct.unpack_from("<Q", payload, offset)[0]
                for offset in range(448, 1_472, 8)
            ],
            dtype=torch.uint8,
        ).reshape(1, MAX_OBSERVERS, MAX_STATES),
    )
    keys = HardFunctorKeys(
        state_keys=_tensor(
            payload[64:192],
            (1, MAX_STATES, 8),
        ),
        action_keys=_tensor(
            payload[192:256],
            (1, MAX_ACTIONS, 8),
        ),
        observer_keys=_tensor(
            payload[256:320],
            (1, MAX_OBSERVERS, 8),
        ),
    )
    keys.validate_masks(machine, 0)
    if machine.deployed_wire(keys, 0) != payload:
        raise DetachedQueryWorkerError(
            "sealed machine wire is noncanonical"
        )
    return machine, keys


@torch.inference_mode()
def execute_detached_query(
    *,
    machine_payload: bytes,
    query_payload: bytes,
    weights_path: Path,
    manifest_path: Path,
    authorization_path: Path,
    expected_authorization_sha256: str,
) -> dict[str, object]:
    """Parse and execute one late query with no source/compiler input."""

    authorization = load_detached_execution_authorization(
        path=authorization_path,
        expected_sha256=expected_authorization_sha256,
    )
    if sha256(machine_payload).hexdigest() != authorization.machine_sha256:
        raise DetachedQueryWorkerError(
            "sealed machine differs from authorization"
        )
    machine, keys = _decode_wire(machine_payload)
    parser, receipt = load_detached_query_parser_package(
        weights_path=weights_path,
        manifest_path=manifest_path,
        expected_manifest_sha256=authorization.parser_manifest_sha256,
    )
    if (
        receipt.state_sha256 != authorization.parser_state_sha256
        or receipt.parameter_count
        != authorization.parser_parameter_count
    ):
        raise DetachedQueryWorkerError(
            "detached parser differs from authorization"
        )
    batch = collate_queries([scan_query(query_payload)])
    parsed = parser(batch, sealed_keys=keys)
    hard_query = parsed.query.harden(machine)
    rollout = execute_hard(machine, hard_query)
    stop = int(hard_query.stop_position[0])
    return {
        "answer": int(rollout.answer[0]),
        "final_state_slot": int(
            rollout.states[0, stop].argmax()
        ),
        "machine_sha256": sha256(machine_payload).hexdigest(),
        "parser_receipt": receipt_payload(receipt),
        "query_parse": {
            "action_slots": [
                int(value)
                for value in hard_query.action_path[0, :stop]
            ],
            "observer_slot": int(hard_query.observer[0]),
            "start_state_slot": int(hard_query.start_state[0]),
            "stop_position": stop,
        },
        "query_sha256": sha256(query_payload).hexdigest(),
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    _assert_detached_import_closure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--parser-weights", type=Path, required=True)
    parser.add_argument("--parser-manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = execute_detached_query(
        machine_payload=arguments.machine.read_bytes(),
        query_payload=arguments.query.read_bytes(),
        weights_path=arguments.parser_weights,
        manifest_path=arguments.parser_manifest,
        authorization_path=arguments.authorization,
        expected_authorization_sha256=(
            arguments.expected_authorization_sha256
        ),
    )
    _write_exclusive(
        arguments.output,
        (
            json.dumps(result, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii"),
    )


if __name__ == "__main__":
    main()
