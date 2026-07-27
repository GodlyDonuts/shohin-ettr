#!/usr/bin/env python3
"""Train-only opcode-complement qualification on fresh-training families."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
from typing import Mapping, Sequence

import torch

from build_er_relation_tensor_board import TRAIN_SPLIT
from er_cst_fresh import canonical_json, derived_seed, trainable_state
from er_dual_stream_fresh_renderers import DualStreamFreshRenderer, render_row
from build_er_dual_stream_fresh_board import validate_row
from er_dual_stream_fresh_scoring import (
    alpha_recode_row,
    distractor_rotate_row,
)
from er_relation_tensor_training import (
    RelationTensorRow,
    byte_batch,
    load_board_receipt,
    load_split,
    parse_row,
)
from er_relation_tensor_adapter import MAX_CARDINALITY, MAX_RULES, MIN_CARDINALITY
from er_dual_stream_relation_adapter import DualStreamRelationCompiler
from pilot_er_cst_rule_card_adapter import state_dict_digest
from pilot_er_dual_stream_fresh import _load_canary, initialize_system
from pilot_er_dual_stream_relation_adapter import EXPECTED_PARAMETERS
from pilot_er_dual_stream_train_canary import (
    fit_train_only,
    score_train_row,
    split_train_families,
)
from pilot_er_relation_tensor import atomic_json_save, atomic_torch_save, release_cuda
from pilot_sd_cst_byte_addressed import sha256_file


SCHEMA = "r12_er_dual_stream_structured_route_canary_v1_5"
EVIDENCE_SCHEMA = "r12_er_dual_stream_structured_route_evidence_v1_5"
REPORT_SCHEMA = "r12_er_dual_stream_structured_route_report_v1_5"
QUALIFIED_CANARY_SEED = 1790361034717866861
BOARD_REPORT_SHA256 = (
    "6b0a011c26c40628cb1db5547715c9f11292cba9af3a9eb10af01714df456b8f"
)
FITTED_ARMS = {
    "legacy_uncoupled": (0.0, False),
    "opcode_coupled": (1.0, False),
    "structured_route": (1.0, True),
}
SEMANTIC_KEYS = (
    "cardinality",
    "initial",
    "relations",
    "rule_active",
    "events",
    "halt",
    "query",
)
ROUTE_PREDICTION_KEYS = (
    "cardinality",
    "initial",
    "relations",
    "rule_active",
    "events",
    "halt",
    "witness_pointer",
    "rule_opcode_pointer",
    "event_opcode_pointer",
)
QUERY_PREDICTION_KEYS = ("query", "query_pointer")
NEUTRAL = re.compile(r"(?<!\S)z[0-9a-z]{5}(?!\S)")
THRESHOLDS = {
    "primary": 0.99,
    "minimum_group": 0.99,
    "identity_deranged_joint_max": 0.10,
    "legacy_joint_max": 0.80,
    "advantage": 0.20,
}
FROZEN_SOURCE_PATHS = (
    "R12_ER_DUAL_STREAM_OPCODE_COUPLED_PREREG.md",
    "train/assess_er_dual_stream_opcode_canary.py",
    "train/er_dual_stream_relation_adapter.py",
    "train/er_relation_tensor_training.py",
    "train/pilot_er_dual_stream_opcode_canary.py",
    "train/pilot_er_dual_stream_fresh.py",
    "train/test_er_dual_stream_opcode_canary.py",
    "train/test_er_dual_stream_relation_adapter.py",
    "train/jobs/er_dual_stream_opcode_canary.sbatch",
)


def source_manifest(repo_root: Path, expected_commit: str) -> dict[str, object]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", *args), cwd=repo_root, capture_output=True, text=True, check=False
        )

    resolved = git("rev-parse", "--verify", f"{expected_commit}^{{commit}}")
    if resolved.returncode or resolved.stdout.strip() != expected_commit:
        raise RuntimeError("opcode-coupled source commit is unavailable")
    hashes = {}
    for relative in FROZEN_SOURCE_PATHS:
        if git("cat-file", "-e", f"{expected_commit}:{relative}").returncode:
            raise RuntimeError(f"opcode-coupled source omits {relative}")
        if git("diff", "--quiet", expected_commit, "--", relative).returncode:
            raise RuntimeError(f"opcode-coupled runtime differs: {relative}")
        hashes[relative] = sha256_file(repo_root / relative)
    value = {"commit": expected_commit, "files": hashes}
    value["sha256"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return value


def _new_neutral(
    family_id: str, label: str, used: set[str], seed: int
) -> str:
    retry = 0
    while True:
        digest = hashlib.sha256(
            f"{seed}:{family_id}:{label}:{retry}".encode()
        ).hexdigest()
        value = "z" + digest[:5]
        if value not in used:
            used.add(value)
            return value
        retry += 1


def renderer_relocation_rows(
    raw_rows: Sequence[Mapping[str, object]],
    probe_family_ids: set[str],
    *,
    seed: int,
) -> list[RelationTensorRow]:
    """Render held-out train semantics in the complementary public coset."""
    representatives: dict[str, Mapping[str, object]] = {}
    for row in raw_rows:
        family = str(row["family_id"])
        if family in probe_family_ids:
            representatives.setdefault(family, row)
    if set(representatives) != probe_family_ids:
        raise ValueError("opcode-coupled relocation families differ")
    output = []
    renderers = tuple(
        DualStreamFreshRenderer(0, witness, 0, query)
        for witness in (0, 1)
        for query in (0, 1)
    )
    for family in sorted(representatives):
        base = representatives[family]
        used = set(NEUTRAL.findall(
            f"{base['program_text']}\n{base['late_query_text']}"
        ))
        rule_noise = [
            _new_neutral(family, f"rule-{slot}", used, seed)
            for slot in range(4)
        ]
        query_noise = _new_neutral(family, "query", used, seed)
        event_noise = _new_neutral(family, "event", used, seed)
        order = list(range(18))
        random.Random(derived_seed(seed, f"{family}:relocate")).shuffle(order)
        event_slot = derived_seed(seed, f"{family}:event-slot") % 13
        for view, renderer in enumerate(renderers):
            relocated = render_row(
                base,
                renderer,
                storage_order=order,
                row_id=f"train-relocate-{family}-v{view}",
                family_id=family,
                rule_distractors=rule_noise,
                event_distractor=event_noise,
                event_distractor_slot=event_slot,
                query_distractor=query_noise,
            )
            validate_row(relocated)
            output.append(parse_row(relocated, TRAIN_SPLIT))
    if len(output) != 4 * len(probe_family_ids):
        raise ValueError("opcode-coupled relocation row count differs")
    return output


def load_raw_train(data_dir: Path, expected_sha256: str) -> list[dict[str, object]]:
    path = data_dir / "train.jsonl"
    if sha256_file(path) != expected_sha256:
        raise ValueError("opcode-coupled raw train hash differs")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 48_000:
        raise ValueError("opcode-coupled raw train row count differs")
    return rows


def relocation_consistency(
    rows: Sequence[RelationTensorRow],
    predictions: Mapping[str, torch.Tensor],
    *,
    keys: Sequence[str] = SEMANTIC_KEYS,
) -> dict[str, object]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.family_id].append(index)
    exact = 0
    for indices in groups.values():
        if len(indices) != 8:
            raise ValueError("opcode-coupled family does not have eight views")
        reference = indices[0]
        exact += int(
            all(
                predictions[key][indices]
                .eq(predictions[key][reference])
                .reshape(len(indices), -1)
                .all()
                for key in keys
            )
        )
    return {"exact": exact, "families": len(groups), "rate": exact / len(groups)}


def prediction_invariance(
    base: Mapping[str, torch.Tensor], changed: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    """Compare retained hard predictions without trusting producer summaries."""
    rows = int(base["pred_cardinality"].shape[0])

    def summary(keys: Sequence[str]) -> dict[str, object]:
        mask = torch.ones(rows, dtype=torch.bool)
        for key in keys:
            left = base[f"pred_{key}"]
            right = changed[f"pred_{key}"]
            if left.shape != right.shape or left.shape[0] != rows:
                raise ValueError(f"invariance prediction shape differs: {key}")
            mask &= left.eq(right).reshape(rows, -1).all(-1)
        exact = int(mask.sum())
        return {"exact": exact, "rows": rows, "rate": exact / rows}

    return {
        "route": summary(ROUTE_PREDICTION_KEYS),
        "query": summary(QUERY_PREDICTION_KEYS),
        "complete": summary((*ROUTE_PREDICTION_KEYS, *QUERY_PREDICTION_KEYS)),
    }


def query_only_alpha_row(row: RelationTensorRow, salt: str) -> RelationTensorRow:
    """Recode neutral query names without changing one program byte."""
    recoded = alpha_recode_row(row, salt)
    return replace(recoded, program_bytes=row.program_bytes)


def query_target_counterfactual_row(
    row: RelationTensorRow, offset: int
) -> RelationTensorRow:
    """Change only the queried slot numeral and scorer-owned query target."""
    if offset <= 0:
        raise ValueError("query counterfactual offset must be positive")
    target = (row.query_position + offset) % row.cardinality
    if target == row.query_position:
        raise ValueError("query counterfactual did not change target")
    start, end = row.query_range
    if end - start != 1:
        raise ValueError("query counterfactual requires one-byte numeral")
    query = bytearray(row.query_bytes)
    query[start:end] = str(target + 1).encode()
    answer = None if row.final_state is None else row.final_state[target]
    return replace(
        row,
        query_bytes=tuple(query),
        query_position=target,
        answer_role=answer,
    )


def identity_deranged_row(row: RelationTensorRow, salt: str) -> RelationTensorRow:
    """Give every neutral source occurrence a unique same-width identity."""
    text = bytes(row.program_bytes).decode("ascii")
    matches = list(NEUTRAL.finditer(text))
    used = set(NEUTRAL.findall(f"{text}\n{bytes(row.query_bytes).decode('ascii')}"))
    seed = derived_seed(QUALIFIED_CANARY_SEED, f"{salt}:{row.row_id}")
    chunks = []
    cursor = 0
    for index, match in enumerate(matches):
        chunks.append(text[cursor : match.start()])
        chunks.append(_new_neutral(row.family_id, f"{salt}-{index}", used, seed))
        cursor = match.end()
    chunks.append(text[cursor:])
    changed = "".join(chunks)
    if len(changed) != len(text) or len(set(NEUTRAL.findall(changed))) != len(matches):
        raise ValueError("identity derangement is not occurrence-unique")
    return replace(row, program_bytes=tuple(changed.encode("ascii")))


def _coherent_route_batch(
    model: DualStreamRelationCompiler,
    rows: Sequence[RelationTensorRow],
    *,
    opcode_weight: float,
    shuffle_opcode: bool = False,
    shuffle_witness: bool = False,
    decoder: str = "coherent",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Decode route pointers with identical downstream relation/event logic."""
    if decoder not in {"coherent", "independent"}:
        raise ValueError(f"unknown structured-route decoder: {decoder}")
    device = next(model.parameters()).device
    program_ids, program_valid = byte_batch(rows, "program_bytes", device)
    query_ids, query_valid = byte_batch(rows, "query_bytes", device)
    output = model.compile_relation_program(
        program_ids, program_valid, query_ids, query_valid
    )
    hard = output.program.hard()
    diagnostics = model.ordered_route_diagnostics(program_ids, program_valid)
    local_logits = diagnostics.local_logits
    if shuffle_witness:
        local_logits = local_logits.clone()
        for row_index in range(local_logits.shape[0]):
            for record in range(local_logits.shape[1]):
                active = diagnostics.candidates[row_index, record]
                for slot in range(local_logits.shape[2]):
                    values = local_logits[row_index, record, slot, active]
                    if values.numel() > 1:
                        local_logits[row_index, record, slot, active] = values.roll(1)
    opcode_logits = diagnostics.opcode_logits
    if shuffle_opcode:
        opcode_logits = opcode_logits.clone()
        for row_index in range(opcode_logits.shape[0]):
            for record in range(opcode_logits.shape[1]):
                active = diagnostics.candidates[row_index, record]
                values = opcode_logits[row_index, record, active]
                if values.numel() > 1:
                    opcode_logits[row_index, record, active] = values.roll(1)
    tables = model._ordered_route_tables(
        local_logits,
        diagnostics.candidates,
        diagnostics.cardinality_logits,
        opcode_logits,
        opcode_weight=opcode_weight,
    )
    batch = len(rows)
    selected_record = diagnostics.routing_assignment.argmax(1)[:, 1 : 1 + MAX_RULES]
    predicted_cardinality = hard.cardinality
    pointers = torch.full(
        (batch, MAX_RULES, 2 * MAX_CARDINALITY),
        -1,
        dtype=torch.long,
        device=device,
    )
    relations = torch.full(
        (batch, MAX_RULES, MAX_CARDINALITY),
        -1,
        dtype=torch.long,
        device=device,
    )
    map_exclusion = torch.full(
        (batch, MAX_RULES), -1, dtype=torch.long, device=device
    )
    rule_opcode_pointer = torch.full_like(map_exclusion, -1)
    event_opcode_pointer = diagnostics.event_opcode_logits.argmax(-1)
    coherent_events = torch.full(
        (batch, len(rows[0].event_halt)), -1, dtype=torch.long, device=device
    )
    candidate_positions = torch.full(
        (batch, MAX_RULES, 13), -1, dtype=torch.long, device=device
    )
    semantic_path_scores = torch.full(
        (batch, MAX_RULES, 4, 13),
        float("-inf"),
        device=device,
    )
    semantic_path_probability = torch.zeros_like(semantic_path_scores)
    semantic_witness_logits = torch.full(
        (batch, MAX_RULES, 2 * MAX_CARDINALITY, 13),
        float("-inf"),
        device=device,
    )
    semantic_opcode_logits = torch.full(
        (batch, MAX_RULES, 13), float("-inf"), device=device
    )
    semantic_marginal_probability = torch.zeros_like(semantic_witness_logits)
    target_exclusion = torch.full_like(map_exclusion, -1)
    path_rank = torch.full_like(map_exclusion, -1)
    correct_path_probability = torch.zeros(
        (batch, MAX_RULES), device=device
    )

    for row_index, row in enumerate(rows):
        cardinality = int(predicted_cardinality[row_index])
        cardinality_index = cardinality - MIN_CARDINALITY
        payload = bytes(row.program_bytes)
        for rule in range(MAX_RULES):
            record = int(selected_record[row_index, rule])
            scores = tables.path_scores[row_index, record]
            probabilities = tables.path_probability[row_index, record]
            semantic_path_scores[row_index, rule] = scores
            semantic_path_probability[row_index, rule] = probabilities
            local = diagnostics.candidates[row_index, record].nonzero().flatten()
            global_positions = diagnostics.source_indices[
                row_index, record, local
            ]
            candidate_positions[
                row_index, rule, : global_positions.numel()
            ] = global_positions
            semantic_witness_logits[
                row_index, rule, :, : local.numel()
            ] = local_logits[row_index, record, :, local]
            semantic_opcode_logits[
                row_index, rule, : local.numel()
            ] = opcode_logits[row_index, record, local]
            semantic_marginal_probability[
                row_index, rule, :, : local.numel()
            ] = tables.marginal_probability[row_index, record, :, local]
            if rule < row.rule_count:
                target_starts = {
                    int(start)
                    for start, _ in (
                        row.witness_before_ranges[rule]
                        + row.witness_after_ranges[rule]
                    )
                }
                target = [
                    rank
                    for rank, position in enumerate(global_positions.tolist())
                    if int(position) not in target_starts
                ]
                if len(target) == 1:
                    target_exclusion[row_index, rule] = target[0]
                    target_index = row.cardinality - MIN_CARDINALITY
                    ranked = scores[target_index].argsort(descending=True)
                    path_rank[row_index, rule] = int(
                        ranked.eq(target[0]).nonzero().flatten()[0]
                    )
                    correct_path_probability[row_index, rule] = probabilities[
                        target_index, target[0]
                    ]
            if (
                not bool(hard.rule_active[row_index, rule])
                or global_positions.numel() != 2 * cardinality + 1
            ):
                continue
            slots = tuple(range(cardinality)) + tuple(
                range(MAX_CARDINALITY, MAX_CARDINALITY + cardinality)
            )
            if decoder == "coherent":
                exclusion = int(scores[cardinality_index].argmax())
                retained = torch.cat(
                    (
                        global_positions[:exclusion],
                        global_positions[exclusion + 1 :],
                    )
                )
            else:
                marginal = tables.marginal_probability[
                    row_index, record, list(slots)
                ]
                selected_local = marginal.argmax(-1)
                retained = diagnostics.source_indices[
                    row_index, record, selected_local
                ]
                retained_set = set(map(int, retained.tolist()))
                candidate_set = set(map(int, global_positions.tolist()))
                missing = candidate_set - retained_set
                if (
                    len(retained_set) != 2 * cardinality
                    or not retained_set <= candidate_set
                    or len(missing) != 1
                ):
                    continue
                excluded_position = missing.pop()
                exclusion = int(
                    global_positions.eq(excluded_position).nonzero().flatten()[0]
                )
            map_exclusion[row_index, rule] = exclusion
            rule_opcode_pointer[row_index, rule] = global_positions[exclusion]
            pointers[row_index, rule, list(slots)] = retained
            before = [payload[int(position) : int(position) + 6] for position in retained[:cardinality]]
            after = [payload[int(position) : int(position) + 6] for position in retained[cardinality:]]
            for after_slot, symbol in enumerate(after):
                matches = [index for index, value in enumerate(before) if value == symbol]
                if len(matches) == 1:
                    relations[row_index, rule, after_slot] = matches[0]

        for event in range(len(row.event_halt)):
            if bool(hard.event_halt[row_index, event]):
                coherent_events[row_index, event] = 0
                continue
            event_position = int(event_opcode_pointer[row_index, event])
            event_symbol = payload[event_position : event_position + 6]
            matches = []
            for rule in range(MAX_RULES):
                rule_position = int(rule_opcode_pointer[row_index, rule])
                if (
                    bool(hard.rule_active[row_index, rule])
                    and rule_position >= 0
                    and payload[rule_position : rule_position + 6] == event_symbol
                ):
                    matches.append(rule)
            if len(matches) == 1:
                coherent_events[row_index, event] = matches[0]

    masked_query = output.query.logits.masked_fill(
        ~hard.active, torch.finfo(output.query.logits.dtype).min
    )
    predictions = {
        "cardinality": hard.cardinality,
        "initial": hard.initial_state.argmax(-1),
        "relations": relations,
        "rule_active": hard.rule_active.long(),
        "events": coherent_events,
        "halt": hard.event_halt.long(),
        "query": masked_query.argmax(-1),
        "witness_pointer": pointers,
        "query_pointer": output.query.pointer_logits.argmax(-1),
        "rule_opcode_pointer": rule_opcode_pointer,
        "event_opcode_pointer": event_opcode_pointer,
    }
    evidence = {
        "selected_record": selected_record,
        "candidate_positions": candidate_positions,
        "path_scores": semantic_path_scores,
        "path_probability": semantic_path_probability,
        "candidate_witness_logits": semantic_witness_logits,
        "candidate_opcode_logits": semantic_opcode_logits,
        "candidate_marginal_probability": semantic_marginal_probability,
        "cardinality_probability": diagnostics.cardinality_logits.float().softmax(-1),
        "query_logits": output.query.logits,
        "query_pointer_logits": output.query.pointer_logits,
        "map_exclusion": map_exclusion,
        "rule_opcode_pointer": rule_opcode_pointer,
        "event_opcode_pointer": event_opcode_pointer,
        "independent_rule_opcode_pointer": diagnostics.rule_opcode_logits.argmax(-1),
        "target_exclusion": target_exclusion,
        "correct_path_rank": path_rank,
        "correct_path_probability": correct_path_probability,
        "target_cardinality": torch.tensor(
            [row.cardinality for row in rows], device=device
        ),
        "target_rule_count": torch.tensor(
            [row.rule_count for row in rows], device=device
        ),
    }
    return (
        {name: value.detach().cpu() for name, value in predictions.items()},
        {name: value.detach().cpu() for name, value in evidence.items()},
    )


def _execute_coherent(
    predictions: Mapping[str, torch.Tensor], row: RelationTensorRow, index: int
) -> tuple[bool, tuple[int, ...], int]:
    cardinality = int(predictions["cardinality"][index])
    if not MIN_CARDINALITY <= cardinality <= MAX_CARDINALITY:
        return False, (), -1
    state = tuple(map(int, predictions["initial"][index, :cardinality]))
    if any(not 0 <= value < cardinality for value in state):
        return False, (), -1
    active = predictions["rule_active"][index].bool()
    alive = True
    for slot in range(len(row.event_halt)):
        if not alive:
            continue
        if bool(predictions["halt"][index, slot]):
            alive = False
            continue
        rule = int(predictions["events"][index, slot])
        if not 0 <= rule < MAX_RULES or not bool(active[rule]):
            return False, (), -1
        relation = tuple(
            map(int, predictions["relations"][index, rule, :cardinality])
        )
        if any(not 0 <= value < cardinality for value in relation):
            return False, (), -1
        state = tuple(state[value] for value in relation)
    query = int(predictions["query"][index])
    if alive or not 0 <= query < cardinality:
        return False, (), -1
    return True, state, state[query]


@torch.no_grad()
def evaluate_coherent_routes(
    model: DualStreamRelationCompiler,
    rows: Sequence[RelationTensorRow],
    *,
    opcode_weight: float,
    batch_size: int,
    shuffle_opcode: bool = False,
    shuffle_witness: bool = False,
    decoder: str = "coherent",
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Score one path decoder and retain sufficient independent evidence."""
    if model.training:
        raise ValueError("structured-route scoring requires evaluation mode")
    fields = defaultdict(list)
    evidence = defaultdict(list)
    predictions_all = defaultdict(list)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        predictions, batch_evidence = _coherent_route_batch(
            model,
            batch,
            opcode_weight=opcode_weight,
            shuffle_opcode=shuffle_opcode,
            shuffle_witness=shuffle_witness,
            decoder=decoder,
        )
        for name, value in predictions.items():
            predictions_all[name].append(value)
        for name, value in batch_evidence.items():
            evidence[name].append(value)
        for index, row in enumerate(batch):
            cardinality = row.cardinality
            target_relations = torch.full(
                (MAX_RULES, MAX_CARDINALITY), -1, dtype=torch.long
            )
            for rule in range(row.rule_count):
                target_relations[rule, :cardinality] = torch.tensor(
                    row.relation_rows[rule]
                )
            exact = {
                "cardinality": int(predictions["cardinality"][index]) == cardinality,
                "initial_rows": torch.equal(
                    predictions["initial"][index, :cardinality],
                    torch.tensor(row.initial_order),
                ),
                "relation_rows": torch.equal(
                    predictions["relations"][index, : row.rule_count, :cardinality],
                    target_relations[: row.rule_count, :cardinality],
                ),
                "rule_active": torch.equal(
                    predictions["rule_active"][index].bool(),
                    torch.tensor(
                        [slot < row.rule_count for slot in range(MAX_RULES)]
                    ),
                ),
                "events": all(
                    bool(halt)
                    or int(predictions["events"][index, slot]) == int(card)
                    for slot, (card, halt) in enumerate(
                        zip(row.event_cards, row.event_halt, strict=True)
                    )
                ),
                "halt": torch.equal(
                    predictions["halt"][index].bool(),
                    torch.tensor(row.event_halt, dtype=torch.bool),
                ),
                "query": int(predictions["query"][index]) == row.query_position,
            }
            witness = True
            for rule in range(row.rule_count):
                ranges = row.witness_before_ranges[rule] + row.witness_after_ranges[rule]
                slots = tuple(range(cardinality)) + tuple(
                    range(MAX_CARDINALITY, MAX_CARDINALITY + cardinality)
                )
                for slot, (low, high) in zip(slots, ranges, strict=True):
                    selected = int(predictions["witness_pointer"][index, rule, slot])
                    witness &= low <= selected < high
            exact["witness_pointer"] = witness
            target_rule_opcode = []
            for rule in range(row.rule_count):
                target_exclusion = int(batch_evidence["target_exclusion"][index, rule])
                if target_exclusion < 0:
                    target_rule_opcode.append(-1)
                else:
                    target_rule_opcode.append(
                        int(
                            batch_evidence["candidate_positions"][
                                index, rule, target_exclusion
                            ]
                        )
                    )
            exact["rule_opcode_pointer"] = all(
                target >= 0
                and int(predictions["rule_opcode_pointer"][index, rule]) == target
                for rule, target in enumerate(target_rule_opcode)
            )
            payload = bytes(row.program_bytes)
            event_opcode_exact = True
            for slot, (card, halt) in enumerate(
                zip(row.event_cards, row.event_halt, strict=True)
            ):
                if bool(halt):
                    continue
                target = target_rule_opcode[card]
                if target < 0:
                    event_opcode_exact = False
                    continue
                symbol = payload[target : target + 6]
                low, high = row.line_ranges[1 + MAX_RULES + slot]
                expected = payload.find(symbol, low, high)
                event_opcode_exact &= (
                    expected >= 0
                    and payload.find(symbol, expected + 1, high) < 0
                    and int(predictions["event_opcode_pointer"][index, slot])
                    == expected
                )
            exact["event_opcode_pointer"] = event_opcode_exact
            exact["query_pointer"] = (
                row.query_range[0]
                <= int(predictions["query_pointer"][index])
                < row.query_range[1]
            )
            packet = all(
                exact[name]
                for name in (
                    "cardinality",
                    "initial_rows",
                    "relation_rows",
                    "rule_active",
                    "events",
                    "halt",
                    "query",
                )
            )
            valid, state, answer = _execute_coherent(predictions, row, index)
            route_packet = all(
                exact[name]
                for name in (
                    "cardinality",
                    "initial_rows",
                    "relation_rows",
                    "rule_active",
                    "events",
                    "halt",
                    "witness_pointer",
                    "rule_opcode_pointer",
                    "event_opcode_pointer",
                )
            )
            exact["packet"] = packet
            exact["state"] = valid and state == row.final_state
            exact["answer"] = valid and answer == row.answer_role
            exact["route_packet"] = route_packet
            exact["route_joint"] = route_packet and exact["state"]
            exact["joint"] = packet and exact["state"] and exact["answer"]
            for name, value in exact.items():
                fields[name].append(bool(value))

    merged_predictions = {
        name: torch.cat(values) for name, values in predictions_all.items()
    }
    merged_evidence = {name: torch.cat(values) for name, values in evidence.items()}
    merged_evidence.update(
        {f"pred_{name}": value for name, value in merged_predictions.items()}
    )

    def summary(values: Sequence[bool]) -> dict[str, object]:
        exact = sum(map(int, values))
        return {"correct": exact, "rows": len(values), "rate": exact / len(values)}

    def grouped(keys: Sequence[object]) -> dict[str, object]:
        return {
            str(key): {
                name: summary(
                    [value for value, item in zip(fields[name], keys, strict=True) if item == key]
                )
                for name in (
                    "packet",
                    "state",
                    "answer",
                    "joint",
                    "route_joint",
                    "witness_pointer",
                    "rule_opcode_pointer",
                    "event_opcode_pointer",
                    "query",
                    "query_pointer",
                )
            }
            for key in sorted(set(keys), key=str)
        }

    result = {
        "overall": {name: summary(values) for name, values in fields.items()},
        "by_cardinality": grouped([row.cardinality for row in rows]),
        "by_renderer": grouped([row.renderer for row in rows]),
        "by_renderer_cardinality": grouped(
            [f"{row.renderer}|N={row.cardinality}" for row in rows]
        ),
    }
    return result, merged_evidence


def _minimum(
    metrics: Mapping[str, object], group: str, fields: Sequence[str]
) -> float:
    values = metrics[group]
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"opcode-coupled {group} is absent")
    return min(
        float(value[field]["rate"])
        for value in values.values()
        for field in fields
    )


def compute_gates(
    arms: Mapping[str, Mapping[str, object]],
    *,
    parameters: Mapping[str, int],
    shared_initialization: bool,
    development_accesses: int,
    confirmation_accesses: int,
) -> tuple[dict[str, bool], dict[str, object]]:
    route_required = (
        "route_joint",
        "state",
        "relation_rows",
        "witness_pointer",
        "rule_opcode_pointer",
        "event_opcode_pointer",
    )
    cell_required = (
        "route_joint",
        "witness_pointer",
        "rule_opcode_pointer",
        "event_opcode_pointer",
    )

    def route_core(arm: str, mode: str) -> bool:
        value = arms[arm]["modes"][mode]
        primary = all(
            float(value[view]["coherent"]["overall"][field]["rate"])
            >= THRESHOLDS["primary"]
            for view in ("canonical", "relocated")
            for field in route_required
        )
        groups = all(
            _minimum(value[view]["coherent"], group, cell_required)
            >= THRESHOLDS["minimum_group"]
            for view in ("canonical", "relocated")
            for group in (
                "by_cardinality",
                "by_renderer",
                "by_renderer_cardinality",
            )
        )
        invariance = all(
            int(value[name]["route"]["exact"])
            == int(value[name]["route"]["rows"])
            == 8_000
            for name in ("alpha", "distractor")
        )
        return (
            primary
            and groups
            and invariance
            and int(value["relocation_consistency"]["exact"])
            == int(value["relocation_consistency"]["families"])
            == 2_000
            and float(
                value["identity_deranged"]["overall"]["route_joint"]["rate"]
            )
            <= THRESHOLDS["identity_deranged_joint_max"]
        )

    def route_joint(arm: str, mode: str, decoder: str = "coherent") -> float:
        return float(
            arms[arm]["modes"][mode]["relocated"][decoder]["overall"][
                "route_joint"
            ]["rate"]
        )

    zero_s0 = route_joint("zero_update", "s0_qraw")
    zero_s0_independent = route_joint("zero_update", "s0_qraw", "independent")
    zero_s1 = route_joint("zero_update", "s1_qraw")
    legacy_s0 = route_joint("legacy_uncoupled", "s0_qraw")
    legacy_s1 = route_joint("legacy_uncoupled", "s1_qraw")
    coupled_s0 = route_joint("opcode_coupled", "s0_qraw")
    coupled_s1 = route_joint("opcode_coupled", "s1_qraw")
    structured_s0 = route_joint("structured_route", "s0_qraw")
    structured_s1 = route_joint("structured_route", "s1_qraw")
    diagnoses = {
        "coherent_decoder_repairs_independent_hardening": route_core(
            "zero_update", "s0_qraw"
        )
        and zero_s0 - zero_s0_independent >= THRESHOLDS["advantage"]
        and zero_s0_independent <= THRESHOLDS["legacy_joint_max"],
        "acute_opcode_coupling_repairs_route": route_core(
            "zero_update", "s1_qraw"
        )
        and zero_s1 - zero_s0 >= THRESHOLDS["advantage"]
        and zero_s0 <= THRESHOLDS["legacy_joint_max"],
        "additional_marginal_training_repairs_route": route_core(
            "legacy_uncoupled", "s0_qraw"
        )
        and legacy_s0 - zero_s0 >= THRESHOLDS["advantage"]
        and zero_s0 <= THRESHOLDS["legacy_joint_max"],
        "legacy_training_plus_acute_opcode_repairs_route": route_core(
            "legacy_uncoupled", "s1_qraw"
        )
        and legacy_s1 - max(zero_s1, legacy_s0) >= THRESHOLDS["advantage"]
        and max(zero_s1, legacy_s0) <= THRESHOLDS["legacy_joint_max"],
        "opcode_coupled_training_repairs_marginals": route_core(
            "opcode_coupled", "s0_qraw"
        )
        and coupled_s0 - max(legacy_s0, zero_s0) >= THRESHOLDS["advantage"]
        and max(legacy_s0, zero_s0) <= THRESHOLDS["legacy_joint_max"],
        "learned_opcode_coupling_repairs_route": route_core(
            "opcode_coupled", "s1_qraw"
        )
        and coupled_s1 - max(legacy_s1, coupled_s0) >= THRESHOLDS["advantage"]
        and max(legacy_s1, coupled_s0) <= THRESHOLDS["legacy_joint_max"],
        "structured_route_nll_repairs_marginals": route_core(
            "structured_route", "s0_qraw"
        )
        and structured_s0 - max(coupled_s0, legacy_s0) >= THRESHOLDS["advantage"]
        and max(coupled_s0, legacy_s0) <= THRESHOLDS["legacy_joint_max"],
        "structured_route_nll_repairs_route": route_core(
            "structured_route", "s1_qraw"
        )
        and structured_s1 - max(structured_s0, coupled_s1, legacy_s1)
        >= THRESHOLDS["advantage"]
        and max(structured_s0, coupled_s1, legacy_s1)
        <= THRESHOLDS["legacy_joint_max"],
    }
    if sum(map(int, diagnoses.values())) != 1:
        selected = None
    elif diagnoses["coherent_decoder_repairs_independent_hardening"]:
        selected = ("zero_update", "s0_qraw", "coherent_decoder")
    elif diagnoses["acute_opcode_coupling_repairs_route"]:
        selected = ("zero_update", "s1_qraw", "acute_opcode_coupling")
    elif diagnoses["additional_marginal_training_repairs_route"]:
        selected = (
            "legacy_uncoupled",
            "s0_qraw",
            "additional_marginal_training",
        )
    elif diagnoses["legacy_training_plus_acute_opcode_repairs_route"]:
        selected = (
            "legacy_uncoupled",
            "s1_qraw",
            "legacy_training_plus_acute_opcode",
        )
    elif diagnoses["opcode_coupled_training_repairs_marginals"]:
        selected = (
            "opcode_coupled",
            "s0_qraw",
            "opcode_coupled_training_marginals",
        )
    elif diagnoses["learned_opcode_coupling_repairs_route"]:
        selected = ("opcode_coupled", "s1_qraw", "learned_opcode_coupling")
    elif diagnoses["structured_route_nll_repairs_marginals"]:
        selected = (
            "structured_route",
            "s0_qraw",
            "structured_route_nll_marginals",
        )
    elif diagnoses["structured_route_nll_repairs_route"]:
        selected = ("structured_route", "s1_qraw", "structured_route_nll")
    else:
        selected = None

    def query_quality(arm: str, query_mode: str) -> tuple[bool, float]:
        score_mode = selected[1].split("_", 1)[0]
        value = arms[arm]["modes"][f"{score_mode}_{query_mode}"]
        rate = min(
            float(value[view]["coherent"]["overall"][field]["rate"])
            for view in ("canonical", "relocated")
            for field in ("query", "query_pointer")
        )
        cell_rate = min(
            float(cell[field]["rate"])
            for view in ("canonical", "relocated")
            for group in (
                "by_cardinality",
                "by_renderer",
                "by_renderer_cardinality",
            )
            for cell in value[view]["coherent"][group].values()
            for field in ("query", "query_pointer")
        )
        recodes = arms[arm]["query_modes"][query_mode]
        counterfactual_rate = min(
            float(recodes[name][view][field]["rate"])
            for name in ("target_a", "target_b")
            for view in ("overall",)
            for field in ("query", "query_pointer")
        )
        counterfactual_cell_rate = min(
            float(cell[field]["rate"])
            for name in ("target_a", "target_b")
            for group in (
                "by_cardinality",
                "by_renderer",
                "by_renderer_cardinality",
            )
            for cell in recodes[name][group].values()
            for field in ("query", "query_pointer")
        )
        invariant = all(
            int(recodes[name]["query"]["exact"])
            == int(recodes[name]["query"]["rows"])
            == 8_000
            for name in ("recode_a", "recode_b")
        ) and all(
            int(value[name]["query"]["exact"])
            == int(value[name]["query"]["rows"])
            == 8_000
            for name in ("alpha", "distractor")
        )
        minimum = min(
            rate,
            cell_rate,
            counterfactual_rate,
            counterfactual_cell_rate,
        )
        return minimum >= THRESHOLDS["primary"] and invariant, minimum

    query_selected = None
    query_rates = None
    if selected is not None:
        raw_ok, raw_rate = query_quality(selected[0], "qraw")
        struct_ok, struct_rate = query_quality(selected[0], "qstruct")
        query_rates = {"qraw": raw_rate, "qstruct": struct_rate}
        if raw_ok:
            query_selected = {"mode": "qraw", "mechanism": "raw_query_grounded"}
        elif (
            struct_ok
            and struct_rate - raw_rate >= THRESHOLDS["advantage"]
            and raw_rate <= THRESHOLDS["legacy_joint_max"]
        ):
            query_selected = {
                "mode": "qstruct",
                "mechanism": "structural_query_repair",
            }

    opcode_causal = True
    if selected is not None and selected[1].startswith("s1"):
        shuffled = float(
            arms[selected[0]]["modes"][selected[1]]["opcode_shuffled"]["overall"][
                "route_joint"
            ]["rate"]
        )
        opcode_causal = (
            route_joint(selected[0], selected[1]) - shuffled
            >= THRESHOLDS["advantage"]
            and shuffled <= THRESHOLDS["legacy_joint_max"]
        )
    witness_causal = False
    if selected is not None:
        shuffled = float(
            arms[selected[0]]["modes"][selected[1]]["witness_shuffled"][
                "overall"
            ]["route_joint"]["rate"]
        )
        witness_causal = (
            route_joint(selected[0], selected[1]) - shuffled
            >= THRESHOLDS["advantage"]
            and shuffled <= THRESHOLDS["legacy_joint_max"]
        )
    gates = {
        "one_uniquely_identified_route_repair_passes": selected is not None,
        "selected_route_repair_passes_all_primary_and_cell_gates": selected
        is not None
        and route_core(selected[0], selected[1]),
        "query_grounding_is_separately_identified": query_selected is not None,
        "opcode_score_is_causal_when_selected": opcode_causal,
        "witness_scores_are_causal": witness_causal,
        "shared_initialization_and_frozen_parent": shared_initialization
        and arms["zero_update"]["fit"]["updates"] == 0
        and all(
            arms[name]["fit"]["frozen_parent_unchanged"] is True
            and int(arms[name]["fit"]["updates"]) == 2_500
            for name in FITTED_ARMS
        ),
        "parameter_certificate_exact_and_below_200m": dict(parameters)
        == EXPECTED_PARAMETERS,
        "train_only_zero_scored_reads": development_accesses == 0
        and confirmation_accesses == 0,
    }
    diagnosis = {
        "classifiers": diagnoses,
        "selected": (
            None
            if selected is None
            else {"arm": selected[0], "mode": selected[1], "mechanism": selected[2]}
        ),
        "query_selected": query_selected,
        "query_rates": query_rates,
        "relocated_route_joint": {
            "zero_s0_independent": zero_s0_independent,
            "zero_s0_coherent": zero_s0,
            "zero_s1_coherent": zero_s1,
            "legacy_s0_coherent": legacy_s0,
            "legacy_s1_coherent": legacy_s1,
            "coupled_s0_coherent": coupled_s0,
            "coupled_s1_coherent": coupled_s1,
            "structured_s0_coherent": structured_s0,
            "structured_s1_coherent": structured_s1,
        },
    }
    return gates, diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--canary-checkpoint", type=Path, required=True)
    for name in (
        "joint_checkpoint",
        "physical_checkpoint",
        "v1_checkpoint",
        "v1_2_checkpoint",
        "confirmed_checkpoint",
        "confirmation_assessment",
        "witness_checkpoint",
        "witness_confirmation_assessment",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing opcode-coupled output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("opcode-coupled canary requires bf16 CUDA")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    board = load_board_receipt(args.data_dir)
    if board.get("report_sha256") != BOARD_REPORT_SHA256:
        raise SystemExit("opcode-coupled board identity differs")
    train_rows = load_split(
        args.data_dir,
        board,
        filename="train.jsonl",
        split=TRAIN_SPLIT,
        expected=48_000,
    )
    canary = _load_canary(args.canary_checkpoint)
    if int(canary.get("seed", -1)) != QUALIFIED_CANARY_SEED:
        raise SystemExit("opcode-coupled qualified-canary seed differs")
    fit_rows, probe_rows, split = split_train_families(
        train_rows,
        derived_seed(QUALIFIED_CANARY_SEED, "dual-stream-train-probe-split"),
    )
    if split != canary.get("split"):
        raise SystemExit("opcode-coupled split differs from qualified canary")
    probe_ids = {row.family_id for row in probe_rows}
    raw = load_raw_train(args.data_dir, str(board["files"]["train.jsonl"]["sha256"]))
    relocated = renderer_relocation_rows(raw, probe_ids, seed=args.seed)
    canonical_scored = [score_train_row(row) for row in probe_rows]
    relocated_scored = [score_train_row(row) for row in relocated]
    combined = canonical_scored + relocated_scored
    device = torch.device("cuda")
    arms: dict[str, dict[str, object]] = {}
    evidence: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "source_manifest": source,
        "seed": args.seed,
        "board_report_sha256": BOARD_REPORT_SHA256,
        "split": split,
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "arms": {},
    }
    initial_digests = set()
    parameters: dict[str, int] | None = None
    arm_weights = {"zero_update": (0.0, False), **FITTED_ARMS}
    for arm_name, (fit_weight, structured_objective) in arm_weights.items():
        model, arm_parameters, frozen_digest, receipt = initialize_system(
            args, device, canary
        )
        parameters = arm_parameters
        model.opcode_coupling_scale = fit_weight
        model.structured_route_objective = structured_objective
        initial_digest = state_dict_digest(trainable_state(model))
        initial_digests.add(initial_digest)
        if arm_name == "zero_update":
            fit = {
                "seed": None,
                "updates": 0,
                "history": [],
                "frozen_parent_unchanged": True,
                "frozen_digest": frozen_digest,
            }
        else:
            fit = fit_train_only(
                model,
                fit_rows,
                seed=derived_seed(args.seed, "opcode-coupled-fit-order"),
                frozen_digest=frozen_digest,
                trainable_names=frozenset(receipt["trainable_names"]),
            )
        model.eval()
        modes: dict[str, dict[str, object]] = {}
        arm_evidence: dict[str, object] = {"modes": {}}
        for score_name, score_weight in (("s0", 0.0), ("s1", 1.0)):
            for query_name, query_structural in (("qraw", False), ("qstruct", True)):
                mode_name = f"{score_name}_{query_name}"
                model.opcode_coupling_scale = score_weight
                model.query_structural_routing = query_structural
                independent_canonical, independent_canonical_evidence = (
                    evaluate_coherent_routes(
                    model,
                    canonical_scored,
                    opcode_weight=score_weight,
                    batch_size=args.batch_size,
                    decoder="independent",
                    )
                )
                independent_relocated, independent_relocated_evidence = (
                    evaluate_coherent_routes(
                    model,
                    relocated_scored,
                    opcode_weight=score_weight,
                    batch_size=args.batch_size,
                    decoder="independent",
                    )
                )
                coherent_canonical, canonical_evidence = evaluate_coherent_routes(
                    model,
                    canonical_scored,
                    opcode_weight=score_weight,
                    batch_size=args.batch_size,
                )
                coherent_relocated, relocated_evidence = evaluate_coherent_routes(
                    model,
                    relocated_scored,
                    opcode_weight=score_weight,
                    batch_size=args.batch_size,
                )
                combined_predictions = {
                    key: torch.cat(
                        (
                            canonical_evidence[f"pred_{key}"],
                            relocated_evidence[f"pred_{key}"],
                        )
                    )
                    for key in SEMANTIC_KEYS
                }
                _, alpha_evidence = evaluate_coherent_routes(
                    model,
                    [
                        alpha_recode_row(row, "opcode-coupled-alpha")
                        for row in relocated_scored
                    ],
                    opcode_weight=score_weight,
                    batch_size=args.batch_size,
                )
                alpha = prediction_invariance(relocated_evidence, alpha_evidence)
                _, distractor_evidence = evaluate_coherent_routes(
                    model,
                    [distractor_rotate_row(row) for row in relocated_scored],
                    opcode_weight=score_weight,
                    batch_size=args.batch_size,
                )
                distractor = prediction_invariance(
                    relocated_evidence, distractor_evidence
                )
                identity_deranged, identity_deranged_evidence = (
                    evaluate_coherent_routes(
                    model,
                    [
                        identity_deranged_row(row, "opcode-coupled-identity")
                        for row in relocated_scored
                    ],
                    opcode_weight=score_weight,
                    batch_size=args.batch_size,
                    )
                )
                opcode_shuffled = None
                opcode_shuffled_evidence = None
                if score_weight:
                    opcode_shuffled, opcode_shuffled_evidence = (
                        evaluate_coherent_routes(
                            model,
                            relocated_scored,
                            opcode_weight=score_weight,
                            batch_size=args.batch_size,
                            shuffle_opcode=True,
                        )
                    )
                witness_shuffled, witness_shuffled_evidence = (
                    evaluate_coherent_routes(
                        model,
                        relocated_scored,
                        opcode_weight=score_weight,
                        batch_size=args.batch_size,
                        shuffle_witness=True,
                    )
                )
                mode = {
                    "score_opcode_weight": score_weight,
                    "query_structural_routing": query_structural,
                    "canonical": {
                        "independent": independent_canonical,
                        "coherent": coherent_canonical,
                    },
                    "relocated": {
                        "independent": independent_relocated,
                        "coherent": coherent_relocated,
                    },
                    "relocation_consistency": relocation_consistency(
                        combined,
                        combined_predictions,
                        keys=tuple(key for key in SEMANTIC_KEYS if key != "query"),
                    ),
                    "alpha": alpha,
                    "distractor": distractor,
                    "identity_deranged": identity_deranged,
                    "opcode_shuffled": opcode_shuffled,
                    "witness_shuffled": witness_shuffled,
                }
                modes[mode_name] = mode

                def compact(value: torch.Tensor) -> torch.Tensor:
                    if value.is_floating_point():
                        return value.to(torch.float16)
                    return value

                arm_evidence["modes"][mode_name] = {
                    "canonical": {
                        "coherent": {
                            key: compact(value)
                            for key, value in canonical_evidence.items()
                        },
                        "independent": {
                            key: compact(value)
                            for key, value in independent_canonical_evidence.items()
                        },
                    },
                    "relocated": {
                        "coherent": {
                            key: compact(value)
                            for key, value in relocated_evidence.items()
                        },
                        "independent": {
                            key: compact(value)
                            for key, value in independent_relocated_evidence.items()
                        },
                    },
                    "alpha": {
                        key: compact(value)
                        for key, value in alpha_evidence.items()
                    },
                    "distractor": {
                        key: compact(value)
                        for key, value in distractor_evidence.items()
                    },
                    "identity_deranged": {
                        key: compact(value)
                        for key, value in identity_deranged_evidence.items()
                    },
                    "opcode_shuffled": None
                    if opcode_shuffled_evidence is None
                    else {
                        key: compact(value)
                        for key, value in opcode_shuffled_evidence.items()
                    },
                    "witness_shuffled": {
                        key: compact(value)
                        for key, value in witness_shuffled_evidence.items()
                    },
                }

        query_modes = {}
        arm_evidence["query_modes"] = {}
        model.opcode_coupling_scale = fit_weight
        for query_name, query_structural in (("qraw", False), ("qstruct", True)):
            model.query_structural_routing = query_structural
            _, base_query_evidence = evaluate_coherent_routes(
                model,
                relocated_scored,
                opcode_weight=fit_weight,
                batch_size=args.batch_size,
            )
            query_modes[query_name] = {}
            arm_evidence["query_modes"][query_name] = {
                "base": {
                    key: compact(value)
                    for key, value in base_query_evidence.items()
                    if key.startswith("pred_")
                    or key in ("query_logits", "query_pointer_logits")
                }
            }
            for label, salt in (
                ("recode_a", "opcode-query-recode-a"),
                ("recode_b", "opcode-query-recode-b"),
            ):
                _, recode_evidence = evaluate_coherent_routes(
                    model,
                    [
                        query_only_alpha_row(row, salt)
                        for row in relocated_scored
                    ],
                    opcode_weight=fit_weight,
                    batch_size=args.batch_size,
                )
                query_modes[query_name][label] = prediction_invariance(
                    base_query_evidence, recode_evidence
                )
                arm_evidence["query_modes"][query_name][label] = {
                    key: compact(value)
                    for key, value in recode_evidence.items()
                    if key.startswith("pred_")
                    or key in ("query_logits", "query_pointer_logits")
                }
            for label, offset in (("target_a", 1), ("target_b", 2)):
                target_metrics, target_evidence = evaluate_coherent_routes(
                    model,
                    [
                        query_target_counterfactual_row(row, offset)
                        for row in relocated_scored
                    ],
                    opcode_weight=fit_weight,
                    batch_size=args.batch_size,
                )
                query_modes[query_name][label] = target_metrics
                arm_evidence["query_modes"][query_name][label] = {
                    key: compact(value) for key, value in target_evidence.items()
                }
        final_trainable_state = trainable_state(model)
        final_trainable_digest = state_dict_digest(final_trainable_state)
        arms[arm_name] = {
            "fit_opcode_coupling_scale": fit_weight,
            "structured_route_objective": structured_objective,
            "initial_state_sha256": initial_digest,
            "fit": fit,
            "modes": modes,
            "query_modes": query_modes,
            "compiler_trainable_state_sha256": final_trainable_digest,
            "compiler_trainable_state": final_trainable_state,
        }
        arm_evidence.update(
            {
                "fit_opcode_coupling_scale": fit_weight,
                "structured_route_objective": structured_objective,
                "initial_state_sha256": initial_digest,
                "fit": fit,
                "compiler_trainable_state_sha256": final_trainable_digest,
            }
        )
        evidence["arms"][arm_name] = arm_evidence
        release_cuda(model)
    if parameters is None:
        raise RuntimeError("opcode-coupled arms are absent")
    gates, diagnosis = compute_gates(
        arms,
        parameters=parameters,
        shared_initialization=len(initial_digests) == 1,
        development_accesses=int(evidence["development_accesses"]),
        confirmation_accesses=int(evidence["confirmation_accesses"]),
    )
    report_arms = {
        name: {key: value for key, value in arm.items() if key != "compiler_trainable_state"}
        for name, arm in arms.items()
    }
    report = {
        "schema": REPORT_SCHEMA,
        "source_manifest": source,
        "seed": args.seed,
        "board_report_sha256": BOARD_REPORT_SHA256,
        "split": split,
        "parameters": parameters,
        "thresholds": THRESHOLDS,
        "arms": report_arms,
        "diagnosis": diagnosis,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "decision": (
            "authorize_new_fresh_board_source"
            if all(gates.values())
            else "reject_opcode_coupled_before_fresh_board"
        ),
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    args.out_dir.mkdir(parents=True)
    atomic_torch_save(
        {
            "schema": SCHEMA,
            "source_manifest": source,
            "seed": args.seed,
            "parameters": parameters,
            "split": split,
            "arms": {
                name: {
                    "fit_opcode_coupling_scale": arm[
                        "fit_opcode_coupling_scale"
                    ],
                    "structured_route_objective": arm[
                        "structured_route_objective"
                    ],
                    "initial_state_sha256": arm["initial_state_sha256"],
                    "fit": arm["fit"],
                    "compiler_trainable_state_sha256": arm[
                        "compiler_trainable_state_sha256"
                    ],
                    "compiler_trainable_state": arm["compiler_trainable_state"],
                }
                for name, arm in arms.items()
            },
            "development_accesses": 0,
            "confirmation_accesses": 0,
        },
        args.out_dir / "compiler.pt",
    )
    atomic_torch_save(evidence, args.out_dir / "train_probe_evidence.pt")
    atomic_json_save(report, args.out_dir / "train_probe_report.json")
    print(json.dumps({"decision": report["decision"], "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
