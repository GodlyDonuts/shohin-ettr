#!/usr/bin/env python3
"""Audit exact ETTR transaction-program diversity without materializing text.

The admitted v3 records already contain independently replayed packets and
transaction traces.  This audit reconstructs the exact 64-step materializer
program for each WORLD x COMMAND corner while deliberately skipping renderer
and QUERY expansion.  It is therefore a CPU-only architecture diagnostic, not
an evaluator and not a source-visible training feature.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ettr_il_v2_materialize import GenericCommand, _build_trace, _project_initial
from ettr_il_v3_materialize import (
    _command_atoms,
    _corner_from_targets,
    _packet_from_value,
)
from ettr_il_v3_protocol import canonical_json_bytes
from materialize_ettr_il_v3_corpus import _iter_records, _sha256_file


REPORT_SCHEMA = "r12-ettr-program-template-audit-v2"
_SPLITS = ("train", "development")
_MODES = ("exact", "structural", "opcode")


class ProgramTemplateAuditError(ValueError):
    """The program-template audit input or output contract differs."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _active_steps(trace: object) -> tuple[tuple[int, ...], ...]:
    names = (
        "opcode",
        "source",
        "target",
        "relation",
        "type_index",
        "value_code",
    )
    columns = [tuple(getattr(trace, name)) for name in names]
    step_mask = tuple(getattr(trace, "step_mask"))
    if not columns or any(len(column) != len(step_mask) for column in columns):
        raise ProgramTemplateAuditError("transaction trace geometry differs")
    if any(type(value) is not bool for value in step_mask):
        raise ProgramTemplateAuditError("transaction step mask differs")
    active = sum(step_mask)
    if step_mask != (True,) * active + (False,) * (len(step_mask) - active):
        raise ProgramTemplateAuditError("transaction step mask is not a prefix")
    return tuple(
        tuple(int(column[index]) for column in columns) for index in range(active)
    )


def trace_signatures(trace: object) -> dict[str, str]:
    """Return exact, operand-structural, and opcode-only program digests."""

    steps = _active_steps(trace)
    return {
        "exact": _digest(steps),
        "structural": _digest(tuple(step[:5] for step in steps)),
        "opcode": _digest(tuple(step[0] for step in steps)),
    }


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total) for count in counter.values()
    )


def _top(counter: Counter[str], limit: int = 32) -> list[dict[str, object]]:
    total = sum(counter.values())
    return [
        {
            "count": count,
            "rate": count / total if total else 0.0,
            "sha256": digest,
        }
        for digest, count in counter.most_common(limit)
    ]


def summarize_counter(counter: Counter[str]) -> dict[str, object]:
    return {
        "entropy_bits": _entropy(counter),
        "instances": sum(counter.values()),
        "top": _top(counter),
        "unique": len(counter),
    }


def _record_programs(record: object) -> Iterable[dict[str, object]]:
    targets = record.assessor_only.targets
    if (
        len(targets.initial_packets) != 2
        or len(targets.terminal_packets) != 4
        or len(targets.transaction_traces) != 4
        or len(targets.answer_matrix) != 4
    ):
        raise ProgramTemplateAuditError("semantic-core target geometry differs")
    initial_packets = tuple(
        _packet_from_value(value, f"initial packet {index}")
        for index, value in enumerate(targets.initial_packets)
    )
    terminal_packets = tuple(
        _packet_from_value(value, f"terminal packet {index}")
        for index, value in enumerate(targets.terminal_packets)
    )
    corners = tuple(
        _corner_from_targets(
            terminal_packets[index],
            targets.transaction_traces[index],
            targets.answer_matrix[index],
            f"corner {index}",
        )
        for index in range(4)
    )
    command_atoms = (
        _command_atoms(terminal_packets[0], "corner 0"),
        _command_atoms(terminal_packets[1], "corner 1"),
    )
    if (
        _command_atoms(terminal_packets[2], "corner 2") != command_atoms[0]
        or _command_atoms(terminal_packets[3], "corner 3") != command_atoms[1]
    ):
        raise ProgramTemplateAuditError("command atoms vary across WORLD")

    family = record.assessor_only.semantic_factors.theory.get("family")
    if type(family) is not str:
        raise ProgramTemplateAuditError("semantic-core family differs")
    commands = record.assessor_only.semantic_factors.commands
    if len(commands) != 2:
        raise ProgramTemplateAuditError("semantic-core command geometry differs")

    for world_index, initial_packet in enumerate(initial_packets):
        initial, static_ranks = _project_initial(
            initial_packet,
            f"world {world_index}",
        )
        for command_index, atoms in enumerate(command_atoms):
            corner_index = 2 * world_index + command_index
            command = GenericCommand(
                sources=(b"audit-a", b"audit-b"),
                command_atoms=atoms,
            )
            terminal, trace = _build_trace(
                initial,
                command,
                corners[corner_index],
                static_ranks,
                f"corner {corner_index}",
            )
            if terminal_packets[corner_index].committed != terminal.committed:
                raise ProgramTemplateAuditError("terminal status differs")
            steps = _active_steps(trace)
            yield {
                **trace_signatures(trace),
                "command": _digest(
                    {
                        "command": commands[command_index],
                        "family": family,
                    }
                ),
                "family": family,
                "opcode_sequence": tuple(step[0] for step in steps),
            }


def _shards(data_root: Path, split: str) -> tuple[Path, ...]:
    root = data_root / split
    if not root.is_dir() or root.is_symlink():
        raise ProgramTemplateAuditError(f"missing immutable split: {split}")
    paths = tuple(sorted(root.glob("*.jsonl.gz")))
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise ProgramTemplateAuditError(f"split shard set differs: {split}")
    return paths


def _audit_shard(
    arguments: tuple[Path, Path, str],
) -> tuple[
    dict[str, Counter[str]],
    Counter[str],
    Counter[str],
    dict[str, tuple[int, ...]],
    set[str],
    dict[str, object],
]:
    path, data_root, split = arguments
    counters = {mode: Counter() for mode in _MODES}
    command_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    opcode_registry: dict[str, tuple[int, ...]] = {}
    core_ids: set[str] = set()
    digest, size = _sha256_file(path)
    rows = 0
    for payload, record in _iter_records(path):
        if record.canonical_bytes() != payload or record.identity.split != split:
            raise ProgramTemplateAuditError("semantic-core record differs")
        if record.identity.core_id in core_ids:
            raise ProgramTemplateAuditError("duplicate semantic-core identity")
        core_ids.add(record.identity.core_id)
        rows += 1
        for program in _record_programs(record):
            for mode in _MODES:
                counters[mode][str(program[mode])] += 1
            command_counter[str(program["command"])] += 1
            family_counter[str(program["family"])] += 1
            opcode_digest = str(program["opcode"])
            opcode_sequence = tuple(program["opcode_sequence"])
            previous = opcode_registry.setdefault(opcode_digest, opcode_sequence)
            if previous != opcode_sequence or _digest(opcode_sequence) != opcode_digest:
                raise ProgramTemplateAuditError("opcode registry collision")
    receipt = {
        "bytes": size,
        "path": path.relative_to(data_root).as_posix(),
        "rows": rows,
        "sha256": digest,
    }
    return (
        counters,
        command_counter,
        family_counter,
        opcode_registry,
        core_ids,
        receipt,
    )


def _audit_split(data_root: Path, split: str, workers: int) -> dict[str, object]:
    if not isinstance(workers, int) or workers < 1:
        raise ProgramTemplateAuditError("worker count differs")
    paths = _shards(data_root, split)
    arguments = tuple((path, data_root, split) for path in paths)
    if workers == 1:
        results = tuple(_audit_shard(argument) for argument in arguments)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            results = tuple(pool.map(_audit_shard, arguments))

    counters = {mode: Counter() for mode in _MODES}
    command_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    opcode_registry: dict[str, tuple[int, ...]] = {}
    core_ids: set[str] = set()
    shard_receipts = []
    for (
        shard_counters,
        commands,
        families,
        shard_registry,
        shard_ids,
        receipt,
    ) in results:
        if core_ids.intersection(shard_ids):
            raise ProgramTemplateAuditError("duplicate semantic-core identity")
        core_ids.update(shard_ids)
        for mode in _MODES:
            counters[mode].update(shard_counters[mode])
        command_counter.update(commands)
        family_counter.update(families)
        for digest, sequence in shard_registry.items():
            previous = opcode_registry.setdefault(digest, sequence)
            if previous != sequence:
                raise ProgramTemplateAuditError("opcode registry collision")
        shard_receipts.append(receipt)
    return {
        "commands": summarize_counter(command_counter),
        "core_rows": len(core_ids),
        "families": dict(sorted(family_counter.items())),
        "opcode_registry": [
            {
                "count": counters["opcode"][digest],
                "opcodes": list(sequence),
                "sha256": digest,
            }
            for digest, sequence in sorted(opcode_registry.items())
        ],
        "programs": {mode: summarize_counter(counters[mode]) for mode in _MODES},
        "shards": shard_receipts,
        "_counters": counters,
        "_command_counter": command_counter,
    }


def _coverage(
    train: Mapping[str, Counter[str]],
    development: Mapping[str, Counter[str]],
) -> dict[str, object]:
    result = {}
    for mode in _MODES:
        train_keys = set(train[mode])
        dev = development[mode]
        seen_instances = sum(count for key, count in dev.items() if key in train_keys)
        seen_unique = sum(key in train_keys for key in dev)
        result[mode] = {
            "development_instance_rate": (
                seen_instances / sum(dev.values()) if dev else 0.0
            ),
            "development_instances_seen": seen_instances,
            "development_unique_rate": seen_unique / len(dev) if dev else 0.0,
            "development_unique_seen": seen_unique,
        }
    return result


def audit(data_root: Path, *, workers: int = 1) -> dict[str, object]:
    data_root = data_root.resolve()
    split_reports = {
        split: _audit_split(data_root, split, workers) for split in _SPLITS
    }
    train_counters = split_reports["train"].pop("_counters")
    development_counters = split_reports["development"].pop("_counters")
    split_reports["train"].pop("_command_counter")
    split_reports["development"].pop("_command_counter")
    report = {
        "cross_split_coverage": _coverage(train_counters, development_counters),
        "data_root": str(data_root),
        "schema": REPORT_SCHEMA,
        "splits": split_reports,
        "status": "pass",
    }
    report["report_payload_sha256"] = _digest(report)
    return report


def _write_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = audit(args.data_root, workers=args.workers)
    _write_no_replace(args.output, canonical_json_bytes(report))
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
