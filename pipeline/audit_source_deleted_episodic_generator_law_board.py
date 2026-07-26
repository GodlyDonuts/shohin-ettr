"""Independent audit for the episodic generator-law reasoning board."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from collections.abc import Mapping, Sequence

from source_deleted_episodic_generator_law_board import (
    DEVELOPMENT_CELLS,
    FAMILIES,
    HELD_OUT_FAMILY,
    HELD_OUT_RENDERER,
    MAX_CLOSURE_DEPTH,
    TRAIN_FAMILIES,
    GeneratedEpisode,
    SealedEpisodicGeneratorMachine,
    build_episode_closure,
    build_frozen_board,
    compile_source,
    compose_support_word,
    decode_query,
    execute_query,
    generate_episode,
    sha256_json,
)


class EpisodicGeneratorLawAuditError(ValueError):
    """Raised when the episodic generator-law board fails a gate."""


_OPAQUE_PATTERN = re.compile(r"h[0-9a-f]{20}")


def _manifest_sha256(
    rows: Sequence[object],
    domain: bytes,
) -> str:
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


def _candidate_manifest(
    row: GeneratedEpisode,
) -> dict[str, str]:
    return {
        "query_sha256": sha256(
            row.candidate.query.encode("ascii")
        ).hexdigest(),
        "source_sha256": sha256(
            row.candidate.source.encode("ascii")
        ).hexdigest(),
    }


def _consistent_maps(
    transitions: Sequence[Sequence[int]],
    observations: Mapping[int, int],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(transition)
        for transition in transitions
        if all(
            transition[source] == target
            for source, target in observations.items()
        )
    )


def _execute_indices(
    transitions: Sequence[Sequence[int]],
    start: int,
    actions: Sequence[int],
) -> int:
    state = start
    for action in actions:
        state = transitions[action][state]
    return state


def _maps_commute(
    first: Sequence[int],
    second: Sequence[int],
) -> bool:
    return all(
        first[second[state]] == second[first[state]]
        for state in range(len(first))
    )


def _cell_contract_holds(row: GeneratedEpisode) -> bool:
    supervisor = row.supervisor
    depths = supervisor.target_composition_lengths
    if supervisor.split == "train":
        return (
            supervisor.cell == "fit"
            and supervisor.family in TRAIN_FAMILIES
            and supervisor.renderer in range(5)
            and supervisor.cardinality == 8
            and depths == (2, 2)
        )
    if supervisor.cell == "law":
        return (
            supervisor.family == HELD_OUT_FAMILY
            and supervisor.renderer == 0
            and supervisor.cardinality == 8
            and depths == (2, 2)
        )
    if supervisor.cell == "composition":
        return (
            supervisor.family in TRAIN_FAMILIES
            and supervisor.renderer == 0
            and supervisor.cardinality == 8
            and min(depths) >= 3
            and max(depths) <= MAX_CLOSURE_DEPTH
        )
    if supervisor.cell == "renderer":
        return (
            supervisor.family in TRAIN_FAMILIES
            and supervisor.renderer == HELD_OUT_RENDERER
            and supervisor.cardinality == 8
            and depths == (2, 2)
        )
    if supervisor.cell == "topology":
        return (
            supervisor.family in TRAIN_FAMILIES
            and supervisor.renderer == 0
            and supervisor.cardinality == 16
            and depths == (2, 2)
        )
    if supervisor.cell == "joint":
        return (
            supervisor.family == HELD_OUT_FAMILY
            and supervisor.renderer == HELD_OUT_RENDERER
            and supervisor.cardinality == 16
            and min(depths) >= 3
            and max(depths) <= MAX_CLOSURE_DEPTH
        )
    return False


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
    source_deletion_passes = 0
    support_key_deletions = 0
    deployed_roundtrips = 0
    compiled_supervisor_matches = 0
    composition_matches = 0
    shortest_word_matches = 0
    unique_target_identifications = 0
    target_visible_records = 0
    target_complete_records = 0
    minimal_witness_records = 0
    removal_ambiguities = 0
    hidden_query_rows = 0
    hidden_query_steps = 0
    visible_query_steps = 0
    law_swap_changes = 0
    action_order_eligible = 0
    action_order_changes = 0
    family_name_leaks = 0
    cell_contract_violations = 0
    target_laws: set[str] = set()
    train_target_laws: set[str] = set()
    development_target_laws: set[str] = set()
    episode_laws: set[str] = set()
    train_support_contexts: set[str] = set()
    development_support_contexts: set[str] = set()
    raw_train_target_maps: set[str] = set()
    raw_development_target_maps: set[str] = set()
    train_target_words: set[tuple[int, ...]] = set()
    development_target_words: list[tuple[int, ...]] = []
    family_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()
    topology_counts: Counter[str] = Counter()
    candidate_manifest: list[object] = []
    supervisor_manifest: list[object] = []

    for row in board:
        supervisor = row.supervisor
        source = row.candidate.source
        source_bytes = source.encode("ascii")
        source_lines = source.splitlines()
        machine = compile_source(source)
        compiled_supervisor_matches += int(
            machine.transition == supervisor.target_transition
            and machine.visible_inputs
            == supervisor.target_visible_inputs
        )
        all_opaque_keys = set(
            _OPAQUE_PATTERN.findall(source)
        )
        support_keys = all_opaque_keys - set(
            machine.target_keys
        )
        deployed_wire = machine.deployed_wire()
        del source
        exact += int(
            execute_query(
                machine,
                row.candidate.query,
            )
            == supervisor.answer
        )
        source_deletion_passes += int(
            source_bytes not in deployed_wire
            and source_bytes not in repr(machine).encode("ascii")
        )
        support_key_deletions += sum(
            key.encode("ascii") not in deployed_wire
            and key.encode("ascii")
            not in repr(machine).encode("ascii")
            for key in support_keys
        )
        deployed_roundtrips += int(
            SealedEpisodicGeneratorMachine.from_deployed_wire(
                deployed_wire
            )
            == machine
        )
        if len(all_opaque_keys) != 4 or len(support_keys) != 2:
            raise EpisodicGeneratorLawAuditError(
                "opaque action role count differs"
            )

        closure = build_episode_closure(
            supervisor.support_transition
        )
        closure_transitions = tuple(
            entry.transition
            for entry in closure
        )
        closure_word_by_map = {
            entry.transition: entry.word
            for entry in closure
        }
        for target_index in range(2):
            target = supervisor.target_transition[target_index]
            word = supervisor.target_composition_words[
                target_index
            ]
            composition_matches += int(
                compose_support_word(
                    supervisor.support_transition,
                    word,
                )
                == target
            )
            shortest_word_matches += int(
                closure_word_by_map.get(target) == word
            )
            visible = set(
                machine.visible_inputs[target_index]
            )
            observations = {
                state: target[state]
                for state in visible
            }
            unique_target_identifications += int(
                _consistent_maps(
                    closure_transitions,
                    observations,
                )
                == (target,)
            )
            for source_state in visible:
                reduced_observations = {
                    state: target[state]
                    for state in visible - {source_state}
                }
                removal_ambiguities += int(
                    len(
                        _consistent_maps(
                            closure_transitions,
                            reduced_observations,
                        )
                    )
                    > 1
                )
            minimal_witness_records += len(visible)

        for line_index in range(1, len(source_lines)):
            if not any(
                target_key in source_lines[line_index]
                for target_key in machine.target_keys
            ):
                continue
            reduced_source = "\n".join(
                line
                for index, line in enumerate(source_lines)
                if index != line_index
            )
            try:
                compile_source(reduced_source)
            except ValueError:
                pass
            else:
                raise EpisodicGeneratorLawAuditError(
                    "target record removal remained sealable"
                )

        start, actions = decode_query(
            machine,
            row.candidate.query,
        )
        state = start
        row_hidden = True
        for action in actions:
            hidden = (
                state not in machine.visible_inputs[action]
            )
            hidden_query_steps += int(hidden)
            visible_query_steps += int(not hidden)
            row_hidden &= hidden
            state = machine.transition[action][state]
        hidden_query_rows += int(row_hidden)

        law_swapped = SealedEpisodicGeneratorMachine(
            cardinality=machine.cardinality,
            target_keys=machine.target_keys,
            transition=tuple(reversed(machine.transition)),
            visible_inputs=machine.visible_inputs,
        )
        law_swap_changes += int(
            execute_query(
                law_swapped,
                row.candidate.query,
            )
            != supervisor.answer
        )
        order_sensitive = not _maps_commute(
            *machine.transition
        )
        if order_sensitive:
            action_order_eligible += 1
            action_order_changes += int(
                _execute_indices(
                    machine.transition,
                    start,
                    tuple(reversed(actions)),
                )
                != supervisor.answer
            )
        if order_sensitive != supervisor.order_sensitive:
            raise EpisodicGeneratorLawAuditError(
                "order-sensitivity label differs"
            )

        target_visible_records += (
            supervisor.target_visible_records
        )
        target_complete_records += (
            supervisor.target_complete_records
        )
        family_name_leaks += int(
            any(
                family in (
                    row.candidate.source
                    + row.candidate.query
                )
                for family in FAMILIES
            )
        )
        cell_contract_violations += int(
            not _cell_contract_holds(row)
        )
        row_target_laws = set(
            supervisor.target_law_sha256
        )
        target_laws.update(row_target_laws)
        episode_laws.add(supervisor.law_sha256)
        support_context = sha256_json(
            {
                "cardinality": supervisor.cardinality,
                "supports": supervisor.support_transition,
            }
        )
        if supervisor.split == "train":
            train_target_laws.update(row_target_laws)
            train_target_words.update(
                supervisor.target_composition_words
            )
            train_support_contexts.add(support_context)
            raw_train_target_maps.update(
                supervisor.target_map_sha256
            )
        else:
            development_target_laws.update(
                row_target_laws
            )
            development_target_words.extend(
                supervisor.target_composition_words
            )
            development_support_contexts.add(
                support_context
            )
            raw_development_target_maps.update(
                supervisor.target_map_sha256
            )
        family_counts[supervisor.family] += 1
        cell_counts[supervisor.cell] += 1
        topology_counts[str(supervisor.cardinality)] += 1
        candidate_manifest.append(
            _candidate_manifest(row)
        )
        supervisor_manifest.append(asdict(supervisor))

    renderer_orbits_total = 0
    renderer_orbits_exact = 0
    renderer_orbits_packet_identical = 0
    renderer_orbits_law_identical = 0
    for family_index, family in enumerate(FAMILIES):
        for cardinality_index, cardinality in enumerate(
            (8, 16)
        ):
            cell = (
                "joint"
                if family == HELD_OUT_FAMILY
                else "composition"
            )
            orbit = [
                generate_episode(
                    seed=(
                        seed
                        + 1_000_000
                        + family_index * 100
                        + cardinality_index
                    ),
                    split="development",
                    family=family,
                    renderer=renderer,
                    cell=cell,
                    cardinality=cardinality,
                )
                for renderer in range(6)
            ]
            machines = [
                compile_source(row.candidate.source)
                for row in orbit
            ]
            renderer_orbits_total += 1
            renderer_orbits_exact += int(
                all(
                    execute_query(
                        machine,
                        row.candidate.query,
                    )
                    == row.supervisor.answer
                    for machine, row in zip(
                        machines,
                        orbit,
                        strict=True,
                    )
                )
            )
            renderer_orbits_packet_identical += int(
                len(
                    {
                        machine.packet_sha256
                        for machine in machines
                    }
                )
                == 1
            )
            renderer_orbits_law_identical += int(
                len(
                    {
                        row.supervisor.law_sha256
                        for row in orbit
                    }
                )
                == 1
            )

    expected_development_cells = set(DEVELOPMENT_CELLS)
    observed_development_cells = {
        row.supervisor.cell
        for row in board
        if row.supervisor.split == "development"
    }
    train_held_out_rows = sum(
        row.supervisor.family == HELD_OUT_FAMILY
        for row in board
        if row.supervisor.split == "train"
    )
    held_out_law_joint_rows = sum(
        row.supervisor.family == HELD_OUT_FAMILY
        and row.supervisor.cell in {"law", "joint"}
        for row in board
        if row.supervisor.split == "development"
    )
    expected_held_out_law_joint_rows = (
        2 * development_per_cell
    )

    if (
        exact != total
        or source_deletion_passes != total
        or support_key_deletions != 2 * total
        or deployed_roundtrips != total
        or compiled_supervisor_matches != total
        or composition_matches != 2 * total
        or shortest_word_matches != 2 * total
        or unique_target_identifications != 2 * total
        or removal_ambiguities != minimal_witness_records
        or hidden_query_rows != total
        or visible_query_steps
        or law_swap_changes != total
        or not action_order_eligible
        or action_order_changes != action_order_eligible
        or family_name_leaks
        or cell_contract_violations
        or len(target_laws) != 2 * total
        or len(episode_laws) != total
        or len(
            raw_train_target_maps
            | raw_development_target_maps
        )
        != 2 * total
        or train_target_laws & development_target_laws
        or raw_train_target_maps
        & raw_development_target_maps
        or train_support_contexts
        & development_support_contexts
        or target_visible_records >= target_complete_records
        or renderer_orbits_exact != renderer_orbits_total
        or renderer_orbits_packet_identical
        != renderer_orbits_total
        or renderer_orbits_law_identical
        != renderer_orbits_total
        or observed_development_cells
        != expected_development_cells
        or train_held_out_rows
        or held_out_law_joint_rows
        != expected_held_out_law_joint_rows
    ):
        raise EpisodicGeneratorLawAuditError(
            "episodic generator-law board audit failed"
        )

    return {
        "action_order_answer_changes": action_order_changes,
        "action_order_eligible": action_order_eligible,
        "board_seed": seed,
        "candidate_manifest_sha256": _manifest_sha256(
            candidate_manifest,
            b"EPISODIC-GENERATOR-LAW-CANDIDATE-V1",
        ),
        "cell_contract_violations": cell_contract_violations,
        "cell_counts": dict(sorted(cell_counts.items())),
        "compiled_supervisor_matches": (
            compiled_supervisor_matches
        ),
        "composition_matches": composition_matches,
        "development_per_cell": development_per_cell,
        "deployed_wire_roundtrips": deployed_roundtrips,
        "development_target_laws": len(
            development_target_laws
        ),
        "development_target_word_instances": len(
            development_target_words
        ),
        "development_target_word_overlap_instances": sum(
            word in train_target_words
            for word in development_target_words
        ),
        "exact_source_deleted": exact,
        "family_counts": dict(sorted(family_counts.items())),
        "family_name_leaks": family_name_leaks,
        "held_out_family": HELD_OUT_FAMILY,
        "held_out_law_joint_rows": held_out_law_joint_rows,
        "hidden_query_rows": hidden_query_rows,
        "hidden_query_steps": hidden_query_steps,
        "law_swap_answer_changes": law_swap_changes,
        "max_closure_depth": MAX_CLOSURE_DEPTH,
        "minimal_witness_records": minimal_witness_records,
        "raw_target_map_overlap": len(
            raw_train_target_maps
            & raw_development_target_maps
        ),
        "record_removals_ambiguous": removal_ambiguities,
        "renderer_orbits_exact": renderer_orbits_exact,
        "renderer_orbits_law_identical": (
            renderer_orbits_law_identical
        ),
        "renderer_orbits_packet_identical": (
            renderer_orbits_packet_identical
        ),
        "renderer_orbits_total": renderer_orbits_total,
        "shortest_word_matches": shortest_word_matches,
        "source_deletion_passes": source_deletion_passes,
        "support_key_deletions": support_key_deletions,
        "supervisor_manifest_sha256": _manifest_sha256(
            supervisor_manifest,
            b"EPISODIC-GENERATOR-LAW-SUPERVISOR-V1",
        ),
        "target_complete_records": target_complete_records,
        "target_law_overlap": len(
            train_target_laws
            & development_target_laws
        ),
        "target_word_holdout_passes": not any(
            word in train_target_words
            for word in development_target_words
        ),
        "target_word_overlap": len(
            train_target_words & set(development_target_words)
        ),
        "target_visible_fraction": (
            target_visible_records / target_complete_records
        ),
        "target_visible_records": target_visible_records,
        "topology_counts": dict(
            sorted(topology_counts.items())
        ),
        "total_rows": total,
        "train_held_out_family_rows": train_held_out_rows,
        "train_per_renderer": train_per_renderer,
        "train_support_development_support_overlap": len(
            train_support_contexts
            & development_support_contexts
        ),
        "train_target_laws": len(train_target_laws),
        "train_target_words": len(train_target_words),
        "unique_episode_laws": len(episode_laws),
        "unique_target_identifications": (
            unique_target_identifications
        ),
        "unique_target_laws": len(target_laws),
        "visible_query_steps": visible_query_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--train-per-renderer",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--development-per-cell",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    receipt = audit_board(
        seed=args.seed,
        train_per_renderer=args.train_per_renderer,
        development_per_cell=args.development_per_cell,
    )
    payload = json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
    ) + "\n"
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(payload, encoding="ascii")
    print(payload, end="")


if __name__ == "__main__":
    main()
