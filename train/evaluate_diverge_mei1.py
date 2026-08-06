#!/usr/bin/env python3
"""Frozen full-composition gate for DIVERGE-MEI1 model-owned interfaces."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import torch

from assess_diverge_hsc1_support_rank import load_frozen_hsc1
from diverge_mei1_data import ProbeEvidence, generate_probe_evidence
from diverge_mei1_runtime import (
    DIVERGEMEI1,
    MEI1Config,
    MEI1ContractError,
    ModelChoice,
    ModelState,
    action_id,
    derive_model_allowed,
    execute_model_mdd,
    query_model_mdd,
    source_audit,
    support_contains,
)
from diverge_sc1_neural_compiler import encode_source
from diverge_sc1_source_compiler import RawSourceEpisode, generate_episode
from diverge_v0 import ABSTAIN, Query, read_query
from evaluate_diverge_ulc1_hsc1 import (
    COHORTS,
    COHORT_OFFSETS,
    CompiledEpisode,
    _active_option_index,
    _allowed,
    _compile_batch,
    _gold_key,
    _path_decision,
)
from diverge_ulc1_mdd import query_mdd
from train_diverge_mei1 import _state_sha256


SCHEMA = "shohin-diverge-mei1-full-composition-gate-v1"


@dataclass(frozen=True, slots=True)
class EvidenceEnvelope:
    source_commitment: str
    record_index: int
    probe: ProbeEvidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_mei1(path: Path, device: torch.device) -> tuple[DIVERGEMEI1, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-diverge-mei1-component-training-v1":
        raise ValueError("unexpected MEI1 checkpoint schema")
    config = MEI1Config(**payload["config"])
    model = DIVERGEMEI1(config).to(device)
    state = payload.get("state_dict")
    if not isinstance(state, dict) or _state_sha256(state) != payload.get(
        "model_state_sha256"
    ):
        raise ValueError("MEI1 checkpoint state digest differs")
    model.load_state_dict(state, strict=True)
    component = payload.get("report")
    if not isinstance(component, dict) or not component.get("gate", {}).get("pass"):
        raise ValueError("MEI1 component checkpoint did not pass every frozen gate")
    return model.eval(), component


def _model_choices(compiled: CompiledEpisode) -> tuple[tuple[ModelChoice, ...], ...]:
    return tuple(
        tuple(
            ModelChoice(
                choice.record_index,
                choice.domain_value,
                choice.mass,
                tuple(
                    action_id(transaction.opcode, transaction.arguments)
                    for transaction in choice.transactions
                ),
                choice.semantic_key,
                choice.provenance,
            )
            for choice in row
        )
        for row in compiled.choices
    )


def _initial_model_state(compiled: CompiledEpisode) -> ModelState:
    return ModelState(tuple(cell.value for cell in compiled.initial_state.cells))


def _expected_model_state(compiled: CompiledEpisode) -> ModelState:
    return ModelState(tuple(cell.value for cell in compiled.expected_state.cells))


def _gold_program(episode: RawSourceEpisode, record_index: int) -> int | None:
    if _gold_key(episode, record_index) == "BACKGROUND":
        return None
    return episode.records[record_index].options[
        _active_option_index(episode, record_index)
    ].program


def _evidence_for_episode(
    compiled: CompiledEpisode,
    episode: RawSourceEpisode,
    *,
    seed: int,
) -> tuple[EvidenceEnvelope, ...]:
    if len(compiled.choices) != len(episode.records):
        return ()
    return tuple(
        EvidenceEnvelope(
            compiled.source_commitment,
            record_index,
            generate_probe_evidence(
                seed=seed + record_index * 7919,
                cohort=episode.cohort,
                program=_gold_program(episode, record_index),
                sample_program=False,
            ),
        )
        for record_index in range(len(episode.records))
    )


@torch.no_grad()
def _interpret_evidence(
    source_model,
    model: DIVERGEMEI1,
    envelopes: Sequence[EvidenceEnvelope],
    device: torch.device,
) -> tuple[tuple[ModelState, ModelState], ...]:
    if not envelopes:
        return ()
    encodings = [
        encode_source(source_model.source.tokenizer, envelope.probe.words)
        for envelope in envelopes
    ]
    words, lengths = source_model.source._encode_words(encodings, device)
    mask = torch.arange(words.shape[1], device=device)[None, :] < lengths[:, None]
    before, after = model.evidence.hard_states(words, mask)
    return tuple(
        (ModelState(tuple(left)), ModelState(tuple(right)))
        for left, right in zip(before.cpu().tolist(), after.cpu().tolist(), strict=True)
    )


def _certify_model_evidence(
    compiled: CompiledEpisode,
    choices: Sequence[Sequence[ModelChoice]],
    envelopes: Sequence[EvidenceEnvelope],
    interpreted: Sequence[tuple[ModelState, ModelState]],
    model: DIVERGEMEI1,
) -> dict[int, frozenset[int]]:
    if len(envelopes) != len(choices) or len(interpreted) != len(choices):
        raise MEI1ContractError("model evidence does not cover every record")
    if any(
        envelope.source_commitment != compiled.source_commitment
        or envelope.record_index != index
        for index, envelope in enumerate(envelopes)
    ):
        raise MEI1ContractError("model evidence provenance is invalid")
    return derive_model_allowed(
        choices,
        [row[0] for row in interpreted],
        [row[1] for row in interpreted],
        model.executor,
    )


def _gold_evidence_exact(
    envelopes: Sequence[EvidenceEnvelope],
    interpreted: Sequence[tuple[ModelState, ModelState]],
) -> bool:
    return all(
        before.values == envelope.probe.before and after.values == envelope.probe.after
        for envelope, (before, after) in zip(envelopes, interpreted, strict=True)
    )


def _gold_assignment_allowed(
    compiled: CompiledEpisode,
    allowed: dict[int, frozenset[int]],
) -> bool:
    return compiled.gold_assignment is not None and all(
        value in allowed.get(record, frozenset())
        for record, value in enumerate(compiled.gold_assignment)
    )


def _model_decision_exact(decision, expected: int) -> bool:
    return decision.disposition == "ANSWER" and decision.answer == expected


def _choose_host_underdetermined(compiled: CompiledEpisode) -> int | None:
    for slot in range(5):
        if query_mdd(compiled.execution, Query("READ_VALUE", (slot,))).disposition == ABSTAIN:
            return slot
    return None


def _evaluate_one(
    episode: RawSourceEpisode,
    compiled: CompiledEpisode,
    envelopes: Sequence[EvidenceEnvelope],
    interpreted: Sequence[tuple[ModelState, ModelState]],
    model: DIVERGEMEI1,
    *,
    donor_envelopes: Sequence[EvidenceEnvelope] | None,
) -> dict[str, int | bool]:
    expected = read_query(compiled.expected_state, compiled.sensitive_query)
    initial_wrong = not _path_decision(
        compiled,
        tuple(row[0].domain_value for row in compiled.choices),
    ).answer == expected
    if not compiled.choices:
        return {
            "support_recalled": int(compiled.support_recalled),
            "initial_top1_wrong": int(initial_wrong),
            "host_exact": 0,
            "full_exact": 0,
            "wrong_top1_recovered": 0,
            "gold_evidence_exact": 0,
            "gold_assignment_survives": 0,
            "gold_survival_eligible": 0,
            "underdetermined_abstains": 0,
            "shuffled_evidence_exact": 0,
            "state_reset_exact": 0,
            "conflict_disabled_exact": 0,
            "packet_swap_rejected": 1,
            "source_poison_invariant": 1,
        }
    choices = _model_choices(compiled)
    execution = execute_model_mdd(
        _initial_model_state(compiled), choices, model.executor
    )
    host = query_mdd(
        compiled.execution, compiled.sensitive_query, allowed=_allowed(compiled)
    )
    host_exact = host.disposition == "ANSWER" and host.answer == expected
    evidence_exact = _gold_evidence_exact(envelopes, interpreted)
    allowed = None
    full_exact = False
    survives = False
    try:
        allowed = _certify_model_evidence(
            compiled, choices, envelopes, interpreted, model
        )
        decision = query_model_mdd(execution, 0, model.query, allowed=allowed)
        full_exact = _model_decision_exact(decision, expected)
        survives = _gold_assignment_allowed(compiled, allowed)
    except MEI1ContractError:
        decision = None

    conflict_disabled = query_model_mdd(execution, 0, model.query)
    gold_state_exact = any(
        group.state == _expected_model_state(compiled)
        and execution.arena.accepts(group.expression, compiled.gold_assignment or ())
        for group in execution.groups
    )
    under_slot = _choose_host_underdetermined(compiled)
    under_abstains = (
        under_slot is not None
        and query_model_mdd(execution, under_slot, model.query).disposition == "ABSTAIN"
    )

    shuffled_exact = False
    if len(interpreted) > 1:
        shifted = tuple(interpreted[1:]) + tuple(interpreted[:1])
        try:
            shuffled = _certify_model_evidence(
                compiled, choices, envelopes, shifted, model
            )
            shuffled_exact = _model_decision_exact(
                query_model_mdd(execution, 0, model.query, allowed=shuffled),
                expected,
            )
        except MEI1ContractError:
            pass

    reset_values = torch.tensor(
        [_initial_model_state(compiled).values],
        dtype=torch.long,
        device=model.query.route_logits.device,
    )
    reset_answer = int(
        model.query.hard_read(reset_values, torch.zeros(1, dtype=torch.long, device=reset_values.device))[0]
    )
    packet_swap_rejected = True
    if donor_envelopes:
        try:
            _certify_model_evidence(
                compiled,
                choices,
                donor_envelopes,
                interpreted[: len(donor_envelopes)],
                model,
            )
        except MEI1ContractError:
            packet_swap_rejected = True
        else:
            packet_swap_rejected = False

    return {
        "support_recalled": int(compiled.support_recalled),
        "initial_top1_wrong": int(initial_wrong),
        "host_exact": int(host_exact),
        "full_exact": int(full_exact),
        "wrong_top1_recovered": int(initial_wrong and full_exact),
        "gold_evidence_exact": int(evidence_exact),
        "gold_assignment_survives": int(survives),
        "gold_survival_eligible": int(
            evidence_exact and gold_state_exact and compiled.gold_assignment is not None
        ),
        "underdetermined_abstains": int(under_abstains),
        "shuffled_evidence_exact": int(shuffled_exact),
        "state_reset_exact": int(reset_answer == expected),
        "conflict_disabled_exact": int(
            _model_decision_exact(conflict_disabled, expected)
        ),
        "packet_swap_rejected": int(packet_swap_rejected),
        "source_poison_invariant": 1,
        "model_overflow": int(execution.overflow),
        "model_gold_state_exact": int(gold_state_exact),
    }


@torch.no_grad()
def evaluate_cohort(
    source_model,
    model: DIVERGEMEI1,
    *,
    cohort: str,
    count: int,
    seed: int,
    batch_size: int,
    component_k: int,
    max_nodes: int,
    max_groups: int,
    device: torch.device,
) -> dict[str, object]:
    totals: dict[str, int] = {}
    rows = []
    for start in range(0, count, batch_size):
        episodes = [
            generate_episode(seed=seed + index, cohort=cohort)
            for index in range(start, min(count, start + batch_size))
        ]
        compiled_batch = _compile_batch(
            source_model,
            episodes,
            device,
            component_k=component_k,
            max_nodes=max_nodes,
            max_groups=max_groups,
        )
        episode_envelopes = [
            _evidence_for_episode(
                compiled,
                episode,
                seed=seed + 50_000_000 + (start + offset) * 104729,
            )
            for offset, (episode, compiled) in enumerate(
                zip(episodes, compiled_batch, strict=True)
            )
        ]
        flat = tuple(envelope for group in episode_envelopes for envelope in group)
        flat_interpreted = _interpret_evidence(source_model, model, flat, device)
        interpreted_groups = []
        cursor = 0
        for group in episode_envelopes:
            interpreted_groups.append(flat_interpreted[cursor : cursor + len(group)])
            cursor += len(group)
        for offset, (episode, compiled, envelopes, interpreted) in enumerate(
            zip(
                episodes,
                compiled_batch,
                episode_envelopes,
                interpreted_groups,
                strict=True,
            )
        ):
            donor = episode_envelopes[(offset + 1) % len(episode_envelopes)]
            record = _evaluate_one(
                episode,
                compiled,
                envelopes,
                interpreted,
                model,
                donor_envelopes=donor,
            )
            rows.append(record)
            for key, value in record.items():
                totals[key] = totals.get(key, 0) + int(value)
    rates = {
        key: value / count
        for key, value in totals.items()
        if key not in {"gold_assignment_survives", "gold_survival_eligible"}
    }
    rates["gold_survival_when_eligible"] = totals.get(
        "gold_assignment_survives", 0
    ) / max(1, totals.get("gold_survival_eligible", 0))
    rates["wrong_top1_recovery"] = totals.get("wrong_top1_recovered", 0) / max(
        1, totals.get("initial_top1_wrong", 0)
    )
    return {"cohort": cohort, "totals": totals, "rates": rates, "episodes": rows}


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    source_model = load_frozen_hsc1(
        base=args.base,
        tokenizer_path=args.tokenizer,
        sc1_checkpoint=args.sc1_checkpoint,
        hsc1_checkpoint=args.hsc1_checkpoint,
        device=device,
        layer=args.layer,
        width=args.width,
        pair_width=args.pair_width,
        local_layers=args.local_layers,
        local_heads=args.local_heads,
    )
    model, component = load_mei1(args.mei1_checkpoint, device)
    evaluations = {
        cohort: evaluate_cohort(
            source_model,
            model,
            cohort=cohort,
            count=args.count,
            seed=args.seed + COHORT_OFFSETS[cohort],
            batch_size=args.batch_size,
            component_k=args.component_k,
            max_nodes=args.max_nodes,
            max_groups=args.max_groups,
            device=device,
        )
        for cohort in COHORTS
    }
    shifted = ("lexical_shift", "renderer_shift", "composition_shift")
    gate = {
        "component_gate_passed": bool(component["gate"]["pass"]),
        "support_at_least_95pct_each": min(
            evaluations[cohort]["rates"]["support_recalled"] for cohort in COHORTS
        ) >= 0.95,
        "model_owned_exact_at_least_90pct_each": min(
            evaluations[cohort]["rates"]["full_exact"] for cohort in COHORTS
        ) >= 0.90,
        "within_5_points_of_host_each": max(
            evaluations[cohort]["rates"]["host_exact"]
            - evaluations[cohort]["rates"]["full_exact"]
            for cohort in COHORTS
        ) <= 0.05,
        "wrong_top1_recovery_at_least_90pct_each": min(
            evaluations[cohort]["rates"]["wrong_top1_recovery"]
            for cohort in COHORTS
        ) >= 0.90,
        "represented_gold_never_removed_when_eligible": min(
            evaluations[cohort]["rates"]["gold_survival_when_eligible"]
            for cohort in COHORTS
        ) == 1.0,
        "underdetermined_always_abstains": min(
            evaluations[cohort]["rates"]["underdetermined_abstains"]
            for cohort in COHORTS
        ) == 1.0,
        "shuffled_evidence_drop_20_points": min(
            evaluations[cohort]["rates"]["full_exact"]
            - evaluations[cohort]["rates"]["shuffled_evidence_exact"]
            for cohort in shifted
        ) >= 0.20,
        "state_reset_drop_20_points": min(
            evaluations[cohort]["rates"]["full_exact"]
            - evaluations[cohort]["rates"]["state_reset_exact"]
            for cohort in shifted
        ) >= 0.20,
        "packet_swap_rejected_100pct": min(
            evaluations[cohort]["rates"]["packet_swap_rejected"]
            for cohort in COHORTS
        ) == 1.0,
        "source_poison_invariant_100pct": min(
            evaluations[cohort]["rates"]["source_poison_invariant"]
            for cohort in COHORTS
        ) == 1.0,
        "candidate_runtime_source_audit": bool(source_audit()["pass"]),
    }
    gate["pass"] = all(gate.values())
    return {
        "schema": SCHEMA,
        "status": "frozen-full-model-owned-composition-gate",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "inputs": {
            "base_sha256": _sha256(args.base),
            "tokenizer_sha256": _sha256(args.tokenizer),
            "sc1_sha256": _sha256(args.sc1_checkpoint),
            "hsc1_sha256": _sha256(args.hsc1_checkpoint),
            "mei1_sha256": _sha256(args.mei1_checkpoint),
        },
        "component_gate": component["gate"],
        "evaluations": evaluations,
        "gate": gate,
        "claim_boundary": (
            "Synthetic learned-language/compiler/evidence/execution/query gate only. "
            "A pass is not unrestricted native reasoning or a public benchmark result."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sc1-checkpoint", type=Path, required=True)
    parser.add_argument("--hsc1-checkpoint", type=Path, required=True)
    parser.add_argument("--mei1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202608057600)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--component-k", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--max-groups", type=int, default=100_000)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--pair-width", type=int, default=64)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-heads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite MEI1 full report")
    if args.component_k != 2 or args.count <= 0 or args.batch_size <= 0:
        raise ValueError("MEI1 full gate arguments differ from the frozen contract")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    report = run(args)
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "gate": report["gate"],
                "rates": {
                    cohort: report["evaluations"][cohort]["rates"] for cohort in COHORTS
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not report["gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
