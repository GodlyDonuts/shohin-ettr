"""Fresh-process executor for one source-deleted EFC machine wire."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import struct


MACHINE_BYTES = 1536
MACHINE_HASH_OFFSET = 1504
MAX_STATES = 16
MAX_ACTIONS = 8
MAX_OBSERVERS = 8


class SealedWorkerError(ValueError):
    """The sealed wire, late query, or output boundary failed closed."""


def _u16(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<H", payload, offset)[0]


def _u32(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<I", payload, offset)[0]


def _u64(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", payload, offset)[0]


def _active_keys(
    payload: bytes,
    *,
    offset: int,
    count: int,
    maximum: int,
    label: str,
) -> tuple[int, ...]:
    values = tuple(
        _u64(payload, offset + index * 8)
        for index in range(maximum)
    )
    active = values[:count]
    if (
        any(value == 0 for value in active)
        or len(set(active)) != count
        or any(values[count:])
    ):
        raise SealedWorkerError(f"sealed {label} keys are invalid")
    return active


def execute_wire(
    machine: bytes,
    query_payload: bytes,
) -> dict[str, object]:
    if (
        len(machine) != MACHINE_BYTES
        or machine[:8] != b"EFCMACH\0"
        or _u32(machine, 8) != 1
        or _u32(machine, 12) != 64
        or _u32(machine, 16) != MACHINE_BYTES
        or machine[MACHINE_HASH_OFFSET:]
        != sha256(machine[:MACHINE_HASH_OFFSET]).digest()
    ):
        raise SealedWorkerError("sealed machine wire is invalid")
    state_count = _u16(machine, 24)
    action_count = _u16(machine, 26)
    observer_count = _u16(machine, 28)
    if (
        not 1 <= state_count <= MAX_STATES
        or not 1 <= action_count <= MAX_ACTIONS
        or not 1 <= observer_count <= MAX_OBSERVERS
        or _u64(machine, 32) != (1 << state_count) - 1
        or _u64(machine, 40) != (1 << action_count) - 1
        or _u64(machine, 48) != (1 << observer_count) - 1
    ):
        raise SealedWorkerError("sealed machine geometry is invalid")
    state_keys = _active_keys(
        machine,
        offset=64,
        count=state_count,
        maximum=MAX_STATES,
        label="state",
    )
    action_keys = _active_keys(
        machine,
        offset=192,
        count=action_count,
        maximum=MAX_ACTIONS,
        label="action",
    )
    observer_keys = _active_keys(
        machine,
        offset=256,
        count=observer_count,
        maximum=MAX_OBSERVERS,
        label="observer",
    )
    try:
        query = json.loads(query_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedWorkerError("late query JSON is invalid") from exc
    if (
        not isinstance(query, dict)
        or set(query) != {"action_keys", "observer_key", "start_key"}
        or not isinstance(query["start_key"], int)
        or not isinstance(query["observer_key"], int)
        or not isinstance(query["action_keys"], list)
        or any(not isinstance(value, int) for value in query["action_keys"])
        or len(query["action_keys"]) > 32
    ):
        raise SealedWorkerError("late query schema differs")
    state_index = {key: index for index, key in enumerate(state_keys)}
    action_index = {key: index for index, key in enumerate(action_keys)}
    observer_index = {
        key: index for index, key in enumerate(observer_keys)
    }
    try:
        state = state_index[query["start_key"]]
        actions = tuple(
            action_index[value] for value in query["action_keys"]
        )
        observer = observer_index[query["observer_key"]]
    except KeyError as exc:
        raise SealedWorkerError("late query key is absent") from exc
    for action in actions:
        state = machine[320 + action * MAX_STATES + state]
        if state >= state_count:
            raise SealedWorkerError(
                "sealed transition reaches an inactive state"
            )
    answer = _u64(
        machine,
        448 + (observer * MAX_STATES + state) * 8,
    )
    return {
        "answer": answer,
        "final_state_key": state_keys[state],
        "final_state_slot": state,
        "machine_sha256": sha256(machine).hexdigest(),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = execute_wire(
        arguments.machine.read_bytes(),
        arguments.query.read_bytes(),
    )
    payload = (
        json.dumps(result, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    _write_exclusive(arguments.output, payload)


if __name__ == "__main__":
    main()
