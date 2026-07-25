"""Independent audit for the variable-topology source-deleted board."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

from source_deleted_variable_topology_board import (
    FAMILIES,
    GeneratedEpisode,
    SealedVariableMachine,
    build_frozen_board,
    compile_source,
    decode_query,
    execute_query,
    generate_episode,
)


_KEY = re.compile(r"h[0-9a-f]{20}")


class VariableTopologyAuditError(ValueError):
    """Raised when the board fails its independent audit."""


def _manifest_sha256(rows: list[object], domain: bytes) -> str:
    digest = sha256(domain + b"\0")
    for row in rows:
        payload = json.dumps(
            row,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _execute_indices(
    machine: SealedVariableMachine,
    start: int,
    actions: tuple[int, ...],
) -> str:
    state = start
    for action in actions:
        state = machine.transition[action][state]
    return machine.state_keys[state]


def _law_swap(machine: SealedVariableMachine) -> SealedVariableMachine:
    return SealedVariableMachine(
        state_keys=machine.state_keys,
        action_keys=machine.action_keys,
        transition=tuple(
            tuple((*row[1:], row[0]))
            for row in machine.transition
        ),
    )


def _candidate_manifest(row: GeneratedEpisode) -> dict[str, object]:
    return {
        "query_sha256": sha256(
            row.candidate.query.encode("ascii")
        ).hexdigest(),
        "source_sha256": sha256(
            row.candidate.source.encode("ascii")
        ).hexdigest(),
    }


def audit_board(
    *,
    seed: int,
    train_per_renderer: int,
    development_per_cell: int,
) -> dict[str, object]:
    board = build_frozen_board(
        seed=seed,
        train_per_renderer=train_per_renderer,
        development_per_cell=development_per_cell,
    )
    exact = 0
    deletion_passes = 0
    family_leaks = 0
    role_neutral = 0
    incidence_type_separable = 0
    incidence_type_ambiguous = 0
    collision_rows = 0
    collision_equal_incidence = 0
    law_swap_changed = 0
    reversal_eligible = 0
    reversal_changed = 0
    topology_counts: Counter[str] = Counter()
    candidate_manifest: list[object] = []
    supervisor_manifest: list[object] = []
    for row in board:
        source = row.candidate.source
        source_bytes = source.encode("ascii")
        machine = compile_source(source)
        query = row.candidate.query
        answer = row.supervisor.answer
        family_leaks += int(
            any(
                family.encode("ascii")
                in source_bytes + row.candidate.query.encode("ascii")
                for family in FAMILIES
            )
        )
        keys = _KEY.findall(source)
        unique_keys = set(keys)
        role_neutral += int(
            len(unique_keys)
            == row.supervisor.cardinality + row.supervisor.action_count
            and all(key.startswith("h") and len(key) == 21 for key in unique_keys)
        )
        counts = Counter(keys)
        if len(set(counts.values())) == 2:
            incidence_type_separable += 1
        else:
            incidence_type_ambiguous += 1
        if row.supervisor.incidence_collision:
            collision_rows += 1
            collision_equal_incidence += int(len(set(counts.values())) == 1)
        del source
        observed = execute_query(machine, query)
        exact += int(observed == answer)
        deletion_passes += int(
            not hasattr(machine, "source")
            and source_bytes not in repr(machine).encode("ascii")
        )
        topology_counts[
            f"{row.supervisor.cardinality}x{row.supervisor.action_count}"
        ] += 1
        start, actions = decode_query(machine, query)
        swapped_answer = _execute_indices(_law_swap(machine), start, actions)
        law_swap_changed += int(swapped_answer != observed)
        if len(actions) > 1 and actions != tuple(reversed(actions)):
            reversal_eligible += 1
            reversed_answer = _execute_indices(
                machine,
                start,
                tuple(reversed(actions)),
            )
            reversal_changed += int(reversed_answer != observed)
        candidate_manifest.append(_candidate_manifest(row))
        supervisor_manifest.append(asdict(row.supervisor))

    orbit_exact = 0
    orbit_total = 0
    orbit_packet_identical = 0
    topologies = (
        (4, 2),
        (4, 3),
        (8, 3),
        (8, 4),
        (8, 5),
        (16, 3),
        (16, 5),
    )
    for family_index, family in enumerate(FAMILIES):
        for topology_index, (cardinality, action_count) in enumerate(topologies):
            orbit = [
                generate_episode(
                    seed=seed + 1_000_000 + family_index * 100 + topology_index,
                    split="development",
                    family=family,
                    renderer=renderer,
                    cell="renderer",
                    cardinality=cardinality,
                    action_count=action_count,
                )
                for renderer in range(6)
            ]
            machines = [
                compile_source(row.candidate.source)
                for row in orbit
            ]
            orbit_total += 1
            orbit_packet_identical += int(
                len({machine.packet_sha256 for machine in machines}) == 1
            )
            orbit_exact += int(
                all(
                    execute_query(machine, row.candidate.query)
                    == row.supervisor.answer
                    for machine, row in zip(machines, orbit, strict=True)
                )
            )

    total = len(board)
    if (
        exact != total
        or deletion_passes != total
        or family_leaks
        or role_neutral != total
        or collision_equal_incidence != collision_rows
        or incidence_type_ambiguous != collision_rows
        or law_swap_changed * 4 < total * 3
        or reversal_changed * 2 < reversal_eligible
        or orbit_exact != orbit_total
        or orbit_packet_identical != orbit_total
    ):
        raise VariableTopologyAuditError("board audit gate failed")
    return {
        "board_seed": seed,
        "candidate_manifest_sha256": _manifest_sha256(
            candidate_manifest,
            b"VARIABLE-TOPOLOGY-CANDIDATE-MANIFEST-V1",
        ),
        "collision_equal_incidence": collision_equal_incidence,
        "collision_rows": collision_rows,
        "development_per_cell": development_per_cell,
        "exact_source_deleted": exact,
        "family_name_leaks": family_leaks,
        "incidence_type_ambiguous": incidence_type_ambiguous,
        "incidence_type_separable": incidence_type_separable,
        "law_swap_answer_changes": law_swap_changed,
        "renderer_orbits_exact": orbit_exact,
        "renderer_orbits_packet_identical": orbit_packet_identical,
        "renderer_orbits_total": orbit_total,
        "reversal_answer_changes": reversal_changed,
        "reversal_eligible": reversal_eligible,
        "role_neutral_key_rows": role_neutral,
        "source_deletion_passes": deletion_passes,
        "supervisor_manifest_sha256": _manifest_sha256(
            supervisor_manifest,
            b"VARIABLE-TOPOLOGY-SUPERVISOR-MANIFEST-V1",
        ),
        "topology_counts": dict(sorted(topology_counts.items())),
        "total_rows": total,
        "train_per_renderer": train_per_renderer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--train-per-renderer", type=int, default=4)
    parser.add_argument("--development-per-cell", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit_board(
        seed=args.seed,
        train_per_renderer=args.train_per_renderer,
        development_per_cell=args.development_per_cell,
    )
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="ascii")
    print(payload, end="")


if __name__ == "__main__":
    main()
