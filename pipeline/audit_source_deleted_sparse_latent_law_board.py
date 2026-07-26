"""Independent audit for the sparse latent-law induction board."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from source_deleted_sparse_latent_law_board import (
    FAMILIES,
    GeneratedEpisode,
    SealedSparseLawMachine,
    SparseLatentLawBoardError,
    build_frozen_board,
    compile_source,
    decode_query,
    execute_query,
    generate_episode,
    union_hypotheses,
)


class SparseLatentLawAuditError(ValueError):
    """Raised when the sparse-law board fails its frozen gates."""


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


def _candidate_manifest(row: GeneratedEpisode) -> dict[str, object]:
    return {
        "query_sha256": sha256(
            row.candidate.query.encode("ascii")
        ).hexdigest(),
        "source_sha256": sha256(
            row.candidate.source.encode("ascii")
        ).hexdigest(),
    }


def _law_shift(machine: SealedSparseLawMachine) -> SealedSparseLawMachine:
    return SealedSparseLawMachine(
        cardinality=machine.cardinality,
        action_keys=machine.action_keys,
        transition=tuple(
            tuple((*row[1:], row[0]))
            for row in machine.transition
        ),
        visible_inputs=machine.visible_inputs,
    )


def _execute_indices(
    machine: SealedSparseLawMachine,
    start: int,
    actions: tuple[int, ...],
) -> int:
    state = start
    for action in actions:
        state = machine.transition[action][state]
    return state


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
    total = len(board)
    exact = 0
    deletion_passes = 0
    deployed_roundtrips = 0
    family_leaks = 0
    hidden_step_rows = 0
    hidden_steps = 0
    visible_steps = 0
    minimal_witness_records = 0
    removal_nonidentifiable = 0
    law_swap_changed = 0
    reversal_eligible = 0
    reversal_changed = 0
    visible_records = 0
    complete_records = 0
    topology_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    law_hashes: set[str] = set()
    train_action_laws: set[str] = set()
    development_action_laws: set[str] = set()
    candidate_manifest: list[object] = []
    supervisor_manifest: list[object] = []
    for row in board:
        source = row.candidate.source
        source_bytes = source.encode("ascii")
        query = row.candidate.query
        query_bytes = query.encode("ascii")
        machine = compile_source(source)
        observed = execute_query(machine, query)
        exact += int(observed == row.supervisor.answer)
        family_leaks += int(
            any(
                family.encode("ascii") in source_bytes + query_bytes
                for family in FAMILIES
            )
        )
        del source
        deletion_passes += int(
            source_bytes not in machine.deployed_wire()
            and source_bytes not in repr(machine).encode("ascii")
        )
        deployed_roundtrips += int(
            SealedSparseLawMachine.from_deployed_wire(
                machine.deployed_wire()
            )
            == machine
        )
        start, actions = decode_query(machine, query)
        state = start
        row_hidden = True
        for action in actions:
            is_hidden = state not in machine.visible_inputs[action]
            hidden_steps += int(is_hidden)
            visible_steps += int(not is_hidden)
            row_hidden &= is_hidden
            state = machine.transition[action][state]
        hidden_step_rows += int(row_hidden)

        lines = row.candidate.source.splitlines()
        minimal_witness_records += len(lines) - 1
        for record_index in range(1, len(lines)):
            reduced = "\n".join(
                line
                for index, line in enumerate(lines)
                if index != record_index
            )
            try:
                compile_source(reduced)
            except SparseLatentLawBoardError:
                removal_nonidentifiable += 1

        shifted = _law_shift(machine)
        law_swap_changed += int(
            _execute_indices(shifted, start, actions) != observed
        )
        if len(actions) > 1 and actions != tuple(reversed(actions)):
            reversal_eligible += 1
            reversal_changed += int(
                _execute_indices(
                    machine,
                    start,
                    tuple(reversed(actions)),
                )
                != observed
            )
        visible_records += row.supervisor.visible_records
        complete_records += row.supervisor.complete_records
        topology_counts[
            f"{row.supervisor.cardinality}x{row.supervisor.action_count}"
        ] += 1
        family_counts[row.supervisor.family] += 1
        law_hashes.add(row.supervisor.law_sha256)
        (
            train_action_laws
            if row.supervisor.split == "train"
            else development_action_laws
        ).update(row.supervisor.action_law_sha256)
        candidate_manifest.append(_candidate_manifest(row))
        supervisor_manifest.append(asdict(row.supervisor))

    orbit_total = 0
    orbit_exact = 0
    orbit_packet_identical = 0
    for family_index, family in enumerate(FAMILIES):
        for topology_index, (cardinality, action_count) in enumerate(
            ((8, 2), (8, 3), (16, 3), (16, 4))
        ):
            orbit = [
                generate_episode(
                    seed=(
                        seed
                        + 1_000_000
                        + family_index * 100
                        + topology_index
                    ),
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
            orbit_exact += int(
                all(
                    execute_query(machine, row.candidate.query)
                    == row.supervisor.answer
                    for machine, row in zip(
                        machines,
                        orbit,
                        strict=True,
                    )
                )
            )
            orbit_packet_identical += int(
                len(
                    {
                        machine.packet_sha256
                        for machine in machines
                    }
                )
                == 1
            )

    if (
        exact != total
        or deletion_passes != total
        or deployed_roundtrips != total
        or family_leaks
        or hidden_step_rows != total
        or visible_steps
        or removal_nonidentifiable != minimal_witness_records
        or len(law_hashes) != total
        or train_action_laws & development_action_laws
        or visible_records * 2 > complete_records
        or law_swap_changed * 4 < total * 3
        or reversal_changed * 2 < reversal_eligible
        or orbit_exact != orbit_total
        or orbit_packet_identical != orbit_total
    ):
        raise SparseLatentLawAuditError("sparse-law board audit failed")
    return {
        "board_seed": seed,
        "candidate_manifest_sha256": _manifest_sha256(
            candidate_manifest,
            b"SPARSE-LATENT-LAW-CANDIDATE-MANIFEST-V1",
        ),
        "complete_records": complete_records,
        "development_per_cell": development_per_cell,
        "deployed_wire_roundtrips": deployed_roundtrips,
        "exact_source_deleted": exact,
        "family_counts": dict(sorted(family_counts.items())),
        "family_name_leaks": family_leaks,
        "development_action_laws": len(development_action_laws),
        "hidden_query_rows": hidden_step_rows,
        "hidden_query_steps": hidden_steps,
        "hypothesis_counts": {
            str(cardinality): len(union_hypotheses(cardinality))
            for cardinality in (8, 16)
        },
        "law_swap_answer_changes": law_swap_changed,
        "minimal_witness_records": minimal_witness_records,
        "record_removals_nonidentifiable": removal_nonidentifiable,
        "renderer_orbits_exact": orbit_exact,
        "renderer_orbits_packet_identical": orbit_packet_identical,
        "renderer_orbits_total": orbit_total,
        "reversal_answer_changes": reversal_changed,
        "reversal_eligible": reversal_eligible,
        "source_deletion_passes": deletion_passes,
        "supervisor_manifest_sha256": _manifest_sha256(
            supervisor_manifest,
            b"SPARSE-LATENT-LAW-SUPERVISOR-MANIFEST-V1",
        ),
        "topology_counts": dict(sorted(topology_counts.items())),
        "total_rows": total,
        "train_per_renderer": train_per_renderer,
        "train_action_laws": len(train_action_laws),
        "train_development_action_law_overlap": len(
            train_action_laws & development_action_laws
        ),
        "unique_laws": len(law_hashes),
        "visible_fraction": visible_records / complete_records,
        "visible_query_steps": visible_steps,
        "visible_records": visible_records,
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
