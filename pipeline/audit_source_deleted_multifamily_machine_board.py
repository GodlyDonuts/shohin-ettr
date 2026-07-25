"""Independent mechanics and leakage audit for the multi-family board."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from source_deleted_multifamily_machine_board import (
    FAMILIES,
    HELD_OUT_RENDERER,
    RENDERERS,
    SealedTransitionMachine,
    build_frozen_board,
    canonical_json,
    compile_source,
    decode_late_query,
    execute_action_indices,
    execute_late_query,
    family_holdout_folds,
    generate_episode,
    sha256_json,
)


AUDIT_VERSION = "SOURCE-DELETED-MULTIFAMILY-AUDIT-V1"
_ROLE_NEUTRAL_KEY = re.compile(r"h[0-9a-f]{20}\Z")


class MultiFamilyAuditError(ValueError):
    """Raised when the frozen board fails an independent audit gate."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_manifest(rows) -> list[dict[str, Any]]:
    return [
        {
            "query_sha256": row.supervisor.query_sha256,
            "source_sha256": row.supervisor.source_sha256,
        }
        for row in rows
    ]


def _supervisor_manifest(rows) -> list[dict[str, Any]]:
    return [asdict(row.supervisor) for row in rows]


def audit_board(
    *,
    seed: int,
    train_per_renderer: int,
    development_per_cell: int,
    orbit_seeds: int,
) -> dict[str, Any]:
    rows = build_frozen_board(
        seed=seed,
        train_per_renderer=train_per_renderer,
        development_per_cell=development_per_cell,
    )
    if not rows:
        raise MultiFamilyAuditError("frozen board is empty")

    train = [row for row in rows if row.supervisor.split == "train"]
    development = [
        row for row in rows if row.supervisor.split == "development"
    ]
    train_laws = {row.supervisor.law_sha256 for row in train}
    development_laws = {row.supervisor.law_sha256 for row in development}
    if train_laws & development_laws:
        raise MultiFamilyAuditError("train and development laws overlap")
    if len({row.supervisor.source_sha256 for row in rows}) != len(rows):
        raise MultiFamilyAuditError("candidate sources are not unique")
    if len({row.supervisor.query_sha256 for row in rows}) != len(rows):
        raise MultiFamilyAuditError("late queries are not unique")

    exact = 0
    source_delete_passes = 0
    family_name_leaks = 0
    role_neutral_key_passes = 0
    machines: list[SealedTransitionMachine] = []
    for row in rows:
        candidate_bytes = (
            row.candidate.source + "\n" + row.candidate.query
        ).encode("ascii")
        family_name_leaks += sum(
            family.encode("ascii") in candidate_bytes for family in FAMILIES
        )
        machine = compile_source(row.candidate.source)
        machines.append(machine)
        role_neutral_key_passes += int(
            all(
                _ROLE_NEUTRAL_KEY.fullmatch(key) is not None
                for key in (*machine.state_keys, *machine.action_keys)
            )
        )
        answer = execute_late_query(machine, row.candidate.query)
        exact += int(answer == row.supervisor.answer)
        source_delete_passes += int(
            execute_late_query(machine, row.candidate.query)
            == row.supervisor.answer
        )
    if exact != len(rows) or source_delete_passes != len(rows):
        raise MultiFamilyAuditError("exact source-deleted mechanics failed")
    if family_name_leaks:
        raise MultiFamilyAuditError("candidate bytes expose a family label")
    if role_neutral_key_passes != len(rows):
        raise MultiFamilyAuditError("opaque key codec exposes semantic roles")

    by_geometry: dict[tuple[str, str, str, int], list[int]] = defaultdict(list)
    for index, (row, machine) in enumerate(zip(rows, machines, strict=True)):
        key = (
            row.supervisor.family,
            row.supervisor.split,
            row.supervisor.cell,
            len(machine.state_keys),
        )
        by_geometry[key].append(index)

    law_swaps = 0
    law_swap_changes = 0
    for indices in by_geometry.values():
        if len(indices) < 2:
            continue
        for position, index in enumerate(indices):
            other_index = indices[(position + 1) % len(indices)]
            if (
                rows[index].supervisor.law_sha256
                == rows[other_index].supervisor.law_sha256
            ):
                continue
            machine = machines[index]
            other = machines[other_index]
            swapped = SealedTransitionMachine(
                state_keys=machine.state_keys,
                action_keys=machine.action_keys,
                transition=other.transition,
            )
            original = execute_late_query(machine, rows[index].candidate.query)
            counterfactual = execute_late_query(
                swapped,
                rows[index].candidate.query,
            )
            law_swaps += 1
            law_swap_changes += int(original != counterfactual)
    if law_swaps == 0 or law_swap_changes / law_swaps < 0.50:
        raise MultiFamilyAuditError("law-swap intervention is not separating")

    reversible_queries = 0
    order_changes = 0
    for row, machine in zip(rows, machines, strict=True):
        start, actions = decode_late_query(machine, row.candidate.query)
        if len(actions) < 2 or actions == tuple(reversed(actions)):
            continue
        original = execute_action_indices(machine, start, actions)
        reversed_answer = execute_action_indices(
            machine,
            start,
            tuple(reversed(actions)),
        )
        reversible_queries += 1
        order_changes += int(original != reversed_answer)
    if reversible_queries == 0 or order_changes / reversible_queries < 0.25:
        raise MultiFamilyAuditError("composition order is insufficiently causal")

    orbit_count = 0
    orbit_exact = 0
    orbit_packet_equal = 0
    for family_index, family in enumerate(FAMILIES):
        for orbit_index in range(orbit_seeds):
            orbit = [
                generate_episode(
                    seed=seed + 1_000_000 + family_index * orbit_seeds + orbit_index,
                    split="development",
                    family=family,
                    renderer=renderer,
                    cell="renderer",
                )
                for renderer in range(len(RENDERERS))
            ]
            if len({row.supervisor.law_sha256 for row in orbit}) != 1:
                raise MultiFamilyAuditError("renderer orbit changes its law")
            if len({row.supervisor.answer for row in orbit}) != 1:
                raise MultiFamilyAuditError("renderer orbit changes its answer")
            packets = [compile_source(row.candidate.source) for row in orbit]
            orbit_packet_equal += int(
                len({packet.packet_sha256 for packet in packets}) == 1
            )
            orbit_exact += int(
                all(
                    execute_late_query(packet, row.candidate.query)
                    == row.supervisor.answer
                    for row, packet in zip(orbit, packets, strict=True)
                )
            )
            orbit_count += 1
    if orbit_exact != orbit_count or orbit_packet_equal != orbit_count:
        raise MultiFamilyAuditError("renderer orbit does not compile identically")

    family_counts = Counter(row.supervisor.family for row in rows)
    split_counts = Counter(row.supervisor.split for row in rows)
    cell_counts = Counter(row.supervisor.cell for row in rows)
    renderer_counts = Counter(row.supervisor.renderer for row in rows)
    composition_counts = Counter(
        row.supervisor.composition_length for row in rows
    )
    candidate_manifest = _candidate_manifest(rows)
    supervisor_manifest = _supervisor_manifest(rows)
    payload: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "board_seed": seed,
        "candidate_manifest_sha256": sha256_json(candidate_manifest),
        "cell_counts": dict(sorted(cell_counts.items())),
        "composition_length_counts": dict(sorted(composition_counts.items())),
        "development_law_count": len(development_laws),
        "exact_source_deleted": {"correct": exact, "total": len(rows)},
        "family_counts": dict(sorted(family_counts.items())),
        "family_holdout_folds": family_holdout_folds(),
        "family_label_leaks": family_name_leaks,
        "held_out_renderer": HELD_OUT_RENDERER,
        "law_swap_intervention": {
            "changed": law_swap_changes,
            "rate": law_swap_changes / law_swaps,
            "total": law_swaps,
        },
        "order_intervention": {
            "changed": order_changes,
            "rate": order_changes / reversible_queries,
            "total": reversible_queries,
        },
        "renderer_counts": dict(sorted(renderer_counts.items())),
        "renderer_orbits": {
            "exact": orbit_exact,
            "packet_equal": orbit_packet_equal,
            "total": orbit_count,
        },
        "row_count": len(rows),
        "role_neutral_key_passes": role_neutral_key_passes,
        "source_delete_passes": source_delete_passes,
        "split_counts": dict(sorted(split_counts.items())),
        "supervisor_manifest_sha256": sha256_json(supervisor_manifest),
        "train_law_count": len(train_laws),
        "train_renderers": list(range(HELD_OUT_RENDERER)),
    }
    payload["payload_sha256"] = sha256_json(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--train-per-renderer", type=int, default=32)
    parser.add_argument("--development-per-cell", type=int, default=32)
    parser.add_argument("--orbit-seeds", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_board(
        seed=args.seed,
        train_per_renderer=args.train_per_renderer,
        development_per_cell=args.development_per_cell,
        orbit_seeds=args.orbit_seeds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        canonical_json(
            {
                "file_sha256": _file_sha256(args.output),
                "output": str(args.output),
                "payload_sha256": payload["payload_sha256"],
                "rows": payload["row_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
