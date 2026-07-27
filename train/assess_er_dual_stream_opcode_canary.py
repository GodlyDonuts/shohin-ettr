#!/usr/bin/env python3
"""Independently rebuild and verify the structured-route qualification."""

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
from build_er_dual_stream_fresh_board import validate_row
from er_cst_fresh import (
    canonical_json,
    derived_seed,
    load_trainable_state,
    trainable_state,
)
from er_dual_stream_fresh_renderers import DualStreamFreshRenderer, render_row
from er_dual_stream_fresh_scoring import (
    alpha_recode_row,
    distractor_rotate_row,
)
from er_relation_tensor_adapter import MAX_CARDINALITY, MAX_RULES
from er_relation_tensor_training import (
    RelationTensorRow,
    byte_batch,
    load_board_receipt,
    load_split,
    parse_row,
)
from pilot_er_dual_stream_relation_adapter import EXPECTED_PARAMETERS
from pilot_er_dual_stream_fresh import _load_canary, initialize_system
from pilot_er_cst_rule_card_adapter import state_dict_digest
from pilot_er_relation_tensor import atomic_json_save, release_cuda
from pilot_sd_cst_byte_addressed import sha256_file


ASSESSMENT_SCHEMA = "r12_er_dual_stream_structured_route_assessment_v1_5"
QUALIFIED_CANARY_SEED = 1790361034717866861
SCHEMA = "r12_er_dual_stream_structured_route_canary_v1_5"
EVIDENCE_SCHEMA = "r12_er_dual_stream_structured_route_evidence_v1_5"
REPORT_SCHEMA = "r12_er_dual_stream_structured_route_report_v1_5"
BOARD_REPORT_SHA256 = (
    "6b0a011c26c40628cb1db5547715c9f11292cba9af3a9eb10af01714df456b8f"
)
FITTED_ARMS = {
    "legacy_uncoupled": (0.0, False),
    "opcode_coupled": (1.0, False),
    "structured_route": (1.0, True),
}
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
THRESHOLDS = {
    "primary": 0.99,
    "minimum_group": 0.99,
    "identity_deranged_joint_max": 0.10,
    "legacy_joint_max": 0.80,
    "advantage": 0.20,
}
NEUTRAL = re.compile(r"(?<!\S)z[0-9a-z]{5}(?!\S)")
FROZEN_SOURCE_PATHS = {
    "R12_ER_DUAL_STREAM_OPCODE_COUPLED_PREREG.md",
    "train/assess_er_dual_stream_opcode_canary.py",
    "train/er_dual_stream_relation_adapter.py",
    "train/er_relation_tensor_training.py",
    "train/pilot_er_dual_stream_opcode_canary.py",
    "train/pilot_er_dual_stream_fresh.py",
    "train/test_er_dual_stream_opcode_canary.py",
    "train/test_er_dual_stream_relation_adapter.py",
    "train/jobs/er_dual_stream_opcode_canary.sbatch",
}
DECISIVE_FIELDS = (
    "route_joint",
    "state",
    "relation_rows",
    "witness_pointer",
    "rule_opcode_pointer",
    "event_opcode_pointer",
    "query",
    "query_pointer",
)
CELL_FIELDS = (
    "route_joint",
    "witness_pointer",
    "rule_opcode_pointer",
    "event_opcode_pointer",
    "query",
    "query_pointer",
)
ROUTE_SEMANTIC_KEYS = (
    "cardinality",
    "initial",
    "relations",
    "rule_active",
    "events",
    "halt",
)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"structured-route {label} is not a mapping")
    return value


def _source_manifest_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    commit = value.get("commit")
    files = value.get("files")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not isinstance(files, Mapping)
        or set(files) != FROZEN_SOURCE_PATHS
        or any(
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in files.values()
        )
    ):
        return False
    payload = {"commit": commit, "files": dict(files)}
    expected = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return value.get("sha256") == expected


def _load_raw_train(data_dir: Path, expected_sha256: str) -> list[dict[str, object]]:
    path = data_dir / "train.jsonl"
    if sha256_file(path) != expected_sha256:
        raise ValueError("structured-route raw train hash differs")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 48_000:
        raise ValueError("structured-route raw train row count differs")
    return rows


def _new_neutral(family_id: str, label: str, used: set[str], seed: int) -> str:
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


def _renderer_relocation_rows(
    raw_rows: Sequence[Mapping[str, object]],
    probe_family_ids: set[str],
    *,
    seed: int,
) -> list[RelationTensorRow]:
    """Independently rebuild the four complementary-coset views per family."""
    representatives: dict[str, Mapping[str, object]] = {}
    for row in raw_rows:
        family = str(row["family_id"])
        if family in probe_family_ids:
            representatives.setdefault(family, row)
    if set(representatives) != probe_family_ids:
        raise ValueError("structured-route relocation families differ")
    renderers = tuple(
        DualStreamFreshRenderer(0, witness, 0, query)
        for witness in (0, 1)
        for query in (0, 1)
    )
    output = []
    for family in sorted(representatives):
        base = representatives[family]
        used = set(
            NEUTRAL.findall(f"{base['program_text']}\n{base['late_query_text']}")
        )
        rule_noise = [
            _new_neutral(family, f"rule-{slot}", used, seed) for slot in range(4)
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
        raise ValueError("structured-route relocation row count differs")
    return output


def _identity_deranged_row(row: RelationTensorRow, salt: str) -> RelationTensorRow:
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
        raise ValueError("structured-route identity derangement differs")
    return replace(row, program_bytes=tuple(changed.encode("ascii")))


def _query_target_counterfactual_row(
    row: RelationTensorRow, offset: int
) -> RelationTensorRow:
    target = (row.query_position + offset) % row.cardinality
    if target == row.query_position:
        raise ValueError("structured-route query counterfactual did not change")
    start, end = row.query_range
    if end - start != 1:
        raise ValueError("structured-route query target width differs")
    query = bytearray(row.query_bytes)
    query[start:end] = str(target + 1).encode()
    answer = None if row.final_state is None else row.final_state[target]
    return replace(
        row,
        query_bytes=tuple(query),
        query_position=target,
        answer_role=answer,
    )


def _split_train_families(
    rows: Sequence[RelationTensorRow], seed: int
) -> tuple[list[RelationTensorRow], list[RelationTensorRow], dict[str, object]]:
    groups: dict[str, list[RelationTensorRow]] = defaultdict(list)
    for row in rows:
        groups[row.family_id].append(row)
    if len(groups) != 12_000 or any(len(group) != 4 for group in groups.values()):
        raise ValueError("structured-route train family shape differs")

    def key(item: tuple[str, list[RelationTensorRow]]) -> tuple[str, str]:
        family = item[0]
        digest = hashlib.sha256(f"{seed}:split:{family}".encode()).hexdigest()
        return digest, family

    ordered = sorted(groups.items(), key=key)
    fit_groups = ordered[:10_000]
    probe_groups = ordered[10_000:]
    fit_ids = {family for family, _ in fit_groups}
    probe_ids = {family for family, _ in probe_groups}
    if fit_ids & probe_ids or len(fit_ids) != 10_000 or len(probe_ids) != 2_000:
        raise ValueError("structured-route train family split is not disjoint")
    fit_rows = [row for _, group in fit_groups for row in group]
    probe_rows = [row for _, group in probe_groups for row in group]
    receipt = {
        "fit_families": len(fit_ids),
        "probe_families": len(probe_ids),
        "fit_rows": len(fit_rows),
        "probe_rows": len(probe_rows),
        "family_overlap": len(fit_ids & probe_ids),
        "fit_family_sha256": hashlib.sha256(
            "\n".join(sorted(fit_ids)).encode()
        ).hexdigest(),
        "probe_family_sha256": hashlib.sha256(
            "\n".join(sorted(probe_ids)).encode()
        ).hexdigest(),
    }
    return fit_rows, probe_rows, receipt


def _score_train_row(row: RelationTensorRow) -> RelationTensorRow:
    state = tuple(row.initial_order)
    alive = True
    for card, halt in zip(row.event_cards, row.event_halt, strict=True):
        if not alive:
            continue
        if halt:
            alive = False
            continue
        state = tuple(state[index] for index in row.relation_rows[card])
    return replace(
        row,
        final_state=state,
        answer_role=state[row.query_position],
    )


def _predictions(value: Mapping[str, object]) -> dict[str, torch.Tensor]:
    output = {
        str(name): tensor.long()
        for name, tensor in value.items()
        if str(name).startswith("pred_") and isinstance(tensor, torch.Tensor)
    }
    missing = {
        f"pred_{name}"
        for name in (*ROUTE_PREDICTION_KEYS, *QUERY_PREDICTION_KEYS)
    } - set(output)
    if missing:
        raise ValueError(f"structured-route predictions omit {sorted(missing)}")
    return output


def _summary(mask: Sequence[bool]) -> dict[str, object]:
    correct = sum(map(int, mask))
    rows = len(mask)
    return {"correct": correct, "rows": rows, "rate": correct / rows}


def _execute(
    predictions: Mapping[str, torch.Tensor], row: RelationTensorRow, index: int
) -> tuple[bool, tuple[int, ...], int]:
    cardinality = int(predictions["pred_cardinality"][index])
    if not 3 <= cardinality <= 6:
        return False, (), -1
    state = tuple(map(int, predictions["pred_initial"][index, :cardinality]))
    if any(not 0 <= value < cardinality for value in state):
        return False, (), -1
    active = predictions["pred_rule_active"][index].bool()
    alive = True
    for slot in range(len(row.event_halt)):
        if not alive:
            continue
        if bool(predictions["pred_halt"][index, slot]):
            alive = False
            continue
        rule = int(predictions["pred_events"][index, slot])
        if not 0 <= rule < MAX_RULES or not bool(active[rule]):
            return False, (), -1
        relation = tuple(
            map(
                int,
                predictions["pred_relations"][index, rule, :cardinality],
            )
        )
        if any(not 0 <= value < cardinality for value in relation):
            return False, (), -1
        state = tuple(state[value] for value in relation)
    query = int(predictions["pred_query"][index])
    if alive or not 0 <= query < cardinality:
        return False, (), -1
    return True, state, state[query]


def _target_opcode_positions(
    row: RelationTensorRow,
    index: int,
    metadata: Mapping[str, torch.Tensor],
) -> list[int]:
    candidates = metadata["candidate_positions"].long()
    exclusion = metadata["target_exclusion"].long()
    output = []
    for rule in range(row.rule_count):
        selected = int(exclusion[index, rule])
        if not 0 <= selected < 2 * row.cardinality + 1:
            output.append(-1)
            continue
        candidate = candidates[index, rule, : 2 * row.cardinality + 1]
        witness_starts = {
            int(start)
            for start, _ in (
                row.witness_before_ranges[rule] + row.witness_after_ranges[rule]
            )
        }
        target = int(candidate[selected])
        retained = {
            int(value)
            for rank, value in enumerate(candidate.tolist())
            if rank != selected
        }
        if target in witness_starts or retained != witness_starts:
            output.append(-1)
        else:
            output.append(target)
    return output


def score_predictions(
    rows: Sequence[RelationTensorRow],
    raw_predictions: Mapping[str, object],
    metadata: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    predictions = _predictions(raw_predictions)
    if any(value.shape[0] != len(rows) for value in predictions.values()):
        raise ValueError("structured-route prediction row count differs")
    fields: dict[str, list[bool]] = defaultdict(list)
    for index, row in enumerate(rows):
        cardinality = row.cardinality
        target_relations = torch.full(
            (MAX_RULES, MAX_CARDINALITY), -1, dtype=torch.long
        )
        for rule in range(row.rule_count):
            target_relations[rule, :cardinality] = torch.tensor(
                row.relation_rows[rule]
            )
        exact = {
            "cardinality": int(predictions["pred_cardinality"][index])
            == cardinality,
            "initial_rows": torch.equal(
                predictions["pred_initial"][index, :cardinality],
                torch.tensor(row.initial_order),
            ),
            "relation_rows": torch.equal(
                predictions["pred_relations"][index, : row.rule_count, :cardinality],
                target_relations[: row.rule_count, :cardinality],
            ),
            "rule_active": torch.equal(
                predictions["pred_rule_active"][index].bool(),
                torch.tensor(
                    [slot < row.rule_count for slot in range(MAX_RULES)]
                ),
            ),
            "events": all(
                bool(halt)
                or int(predictions["pred_events"][index, slot]) == int(card)
                for slot, (card, halt) in enumerate(
                    zip(row.event_cards, row.event_halt, strict=True)
                )
            ),
            "halt": torch.equal(
                predictions["pred_halt"][index].bool(),
                torch.tensor(row.event_halt, dtype=torch.bool),
            ),
            "query": int(predictions["pred_query"][index])
            == row.query_position,
            "query_pointer": row.query_range[0]
            <= int(predictions["pred_query_pointer"][index])
            < row.query_range[1],
        }
        witness_exact = True
        for rule in range(row.rule_count):
            spans = row.witness_before_ranges[rule] + row.witness_after_ranges[rule]
            slots = tuple(range(cardinality)) + tuple(
                range(MAX_CARDINALITY, MAX_CARDINALITY + cardinality)
            )
            for slot, (low, high) in zip(slots, spans, strict=True):
                selected = int(
                    predictions["pred_witness_pointer"][index, rule, slot]
                )
                witness_exact &= low <= selected < high
        exact["witness_pointer"] = witness_exact
        target_rule_opcode = _target_opcode_positions(row, index, metadata)
        exact["rule_opcode_pointer"] = all(
            target >= 0
            and int(predictions["pred_rule_opcode_pointer"][index, rule])
            == target
            for rule, target in enumerate(target_rule_opcode)
        )
        payload = bytes(row.program_bytes)
        event_exact = True
        for slot, (card, halt) in enumerate(
            zip(row.event_cards, row.event_halt, strict=True)
        ):
            if bool(halt):
                continue
            target = target_rule_opcode[card]
            if target < 0:
                event_exact = False
                continue
            symbol = payload[target : target + 6]
            low, high = row.line_ranges[1 + MAX_RULES + slot]
            expected = payload.find(symbol, low, high)
            event_exact &= (
                expected >= 0
                and payload.find(symbol, expected + 1, high) < 0
                and int(predictions["pred_event_opcode_pointer"][index, slot])
                == expected
            )
        exact["event_opcode_pointer"] = event_exact
        valid, state, answer = _execute(predictions, row, index)
        exact["state"] = valid and state == row.final_state
        exact["answer"] = valid and answer == row.answer_role
        exact["route_joint"] = all(
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
                "state",
            )
        )
        for name, value in exact.items():
            fields[name].append(bool(value))

    def grouped(keys: Sequence[object]) -> dict[str, object]:
        return {
            str(key): {
                field: _summary(
                    [
                        value
                        for value, item in zip(fields[field], keys, strict=True)
                        if item == key
                    ]
                )
                for field in CELL_FIELDS
            }
            for key in sorted(set(keys), key=str)
        }

    return {
        "overall": {name: _summary(values) for name, values in fields.items()},
        "by_cardinality": grouped([row.cardinality for row in rows]),
        "by_renderer": grouped([row.renderer for row in rows]),
        "by_renderer_cardinality": grouped(
            [f"{row.renderer}|N={row.cardinality}" for row in rows]
        ),
    }


def prediction_invariance(
    base_raw: Mapping[str, object], changed_raw: Mapping[str, object]
) -> dict[str, object]:
    base = _predictions(base_raw)
    changed = _predictions(changed_raw)
    rows = int(base["pred_cardinality"].shape[0])

    def exact(keys: Sequence[str]) -> dict[str, object]:
        mask = torch.ones(rows, dtype=torch.bool)
        for key in keys:
            left = base[f"pred_{key}"]
            right = changed[f"pred_{key}"]
            if left.shape != right.shape:
                raise ValueError(f"structured-route invariance shape differs: {key}")
            mask &= left.eq(right).reshape(rows, -1).all(-1)
        return _summary(mask.tolist())

    return {
        "route": exact(ROUTE_PREDICTION_KEYS),
        "query": exact(QUERY_PREDICTION_KEYS),
        "complete": exact((*ROUTE_PREDICTION_KEYS, *QUERY_PREDICTION_KEYS)),
    }


def verify_query_logit_evidence(evidence: Mapping[str, object]) -> dict[str, object]:
    checked = semantic_exact = pointer_exact = finite = 0

    def visit(value: object) -> None:
        nonlocal checked, semantic_exact, pointer_exact, finite
        if not isinstance(value, Mapping):
            return
        if "pred_query" in value:
            required = (
                "pred_cardinality",
                "pred_query_pointer",
                "query_logits",
                "query_pointer_logits",
            )
            if any(name not in value for name in required):
                return
            query = value["pred_query"].long()
            cardinality = value["pred_cardinality"].long()
            pointer = value["pred_query_pointer"].long()
            query_logits = value["query_logits"].float()
            pointer_logits = value["query_pointer_logits"].float()
            if (
                query.ndim != 1
                or cardinality.shape != query.shape
                or pointer.shape != query.shape
                or query_logits.shape[0] != query.shape[0]
                or pointer_logits.shape[0] != query.shape[0]
            ):
                return
            active = torch.arange(query_logits.shape[-1])[None] < cardinality[:, None]
            masked = query_logits.masked_fill(~active, torch.finfo(torch.float32).min)
            checked += 1
            semantic_exact += int(torch.equal(masked.argmax(-1), query))
            pointer_exact += int(torch.equal(pointer_logits.argmax(-1), pointer))
            finite += int(
                bool(torch.isfinite(query_logits).all())
                and bool(torch.isfinite(pointer_logits).all())
            )
        for nested in value.values():
            if isinstance(nested, Mapping):
                visit(nested)

    visit(evidence)
    return {
        "branches_checked": checked,
        "all_query_semantic_argmax_exact": checked > 0 and semantic_exact == checked,
        "all_query_pointer_argmax_exact": checked > 0 and pointer_exact == checked,
        "all_query_logits_finite": checked > 0 and finite == checked,
    }


def relocation_consistency(
    canonical_rows: Sequence[RelationTensorRow],
    relocated_rows: Sequence[RelationTensorRow],
    canonical_raw: Mapping[str, object],
    relocated_raw: Mapping[str, object],
) -> dict[str, object]:
    rows = [*canonical_rows, *relocated_rows]
    left = _predictions(canonical_raw)
    right = _predictions(relocated_raw)
    predictions = {
        name: torch.cat((left[name], right[name])) for name in left
    }
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.family_id].append(index)
    exact = 0
    for indices in groups.values():
        if len(indices) != 8:
            raise ValueError("structured-route relocation family count differs")
        reference = indices[0]
        exact += int(
            all(
                predictions[f"pred_{key}"][indices]
                .eq(predictions[f"pred_{key}"][reference])
                .reshape(len(indices), -1)
                .all()
                for key in ROUTE_SEMANTIC_KEYS
            )
        )
    return {"exact": exact, "families": len(groups), "rate": exact / len(groups)}


def _metrics_equal(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    for field in DECISIVE_FIELDS:
        if _mapping(left["overall"], "left overall").get(field) != _mapping(
            right["overall"], "right overall"
        ).get(field):
            return False
    for group in (
        "by_cardinality",
        "by_renderer",
        "by_renderer_cardinality",
    ):
        left_group = _mapping(left[group], f"left {group}")
        right_group = _mapping(right[group], f"right {group}")
        if set(left_group) != set(right_group):
            return False
        for key in left_group:
            left_cell = _mapping(left_group[key], f"left {group} cell")
            right_cell = _mapping(right_group[key], f"right {group} cell")
            if any(left_cell[field] != right_cell.get(field) for field in CELL_FIELDS):
                return False
    return True


def verify_path_evidence(
    evidence: Mapping[str, object],
    rows: Sequence[RelationTensorRow] | None = None,
    *,
    view_name: str = "relocated",
    verify_controls: bool = True,
) -> dict[str, object]:
    checked = map_exact = complement_exact = probability_exact = softmax_exact = 0
    path_score_exact = marginal_exact = independent_exact = 0
    opcode_rotation_exact = witness_rotation_exact = intervention_decode_exact = 0
    opcode_rotation_checked = witness_rotation_checked = 0
    for arm_name, arm_raw in _mapping(evidence["arms"], "arms").items():
        modes = _mapping(_mapping(arm_raw, f"arm {arm_name}")["modes"], "modes")
        for mode_name in ("s0_qraw", "s1_qraw", "s0_qstruct", "s1_qstruct"):
            mode = _mapping(modes[mode_name], mode_name)
            relocated_view = _mapping(mode[view_name], view_name)
            relocated = _mapping(relocated_view["coherent"], "coherent")
            independent = _mapping(
                relocated_view["independent"], "independent"
            )
            scores = relocated["path_scores"].float()
            probabilities = relocated["path_probability"].float()
            predicted_cardinality = relocated["pred_cardinality"].long()
            target_cardinality = relocated["target_cardinality"].long()
            target_rule_count = relocated["target_rule_count"].long()
            map_exclusion = relocated["map_exclusion"].long()
            candidates = relocated["candidate_positions"].long()
            witness = relocated["pred_witness_pointer"].long()
            opcode = relocated["rule_opcode_pointer"].long()
            witness_logits = relocated["candidate_witness_logits"].float()
            opcode_logits = relocated["candidate_opcode_logits"].float()
            marginal = relocated["candidate_marginal_probability"].float()
            cardinality_probability = relocated["cardinality_probability"].float()
            expected_rows = 8_000 if rows is None else len(rows)
            if scores.shape != (expected_rows, 4, 4, 13) or probabilities.shape != scores.shape:
                raise ValueError("structured-route path evidence shape differs")
            if cardinality_probability.shape != (expected_rows, 4):
                raise ValueError("structured-route cardinality probability shape differs")
            if rows is not None:
                expected_cardinality = torch.tensor([item.cardinality for item in rows])
                expected_rule_count = torch.tensor([item.rule_count for item in rows])
                if not torch.equal(target_cardinality, expected_cardinality) or not torch.equal(
                    target_rule_count, expected_rule_count
                ):
                    raise ValueError("structured-route retained targets differ from board")
            weight = float(mode_name.startswith("s1"))
            controls = {
                "opcode": mode.get("opcode_shuffled"),
                "witness": mode.get("witness_shuffled"),
            } if verify_controls else {}
            for row in range(expected_rows):
                cardinality = int(predicted_cardinality[row])
                if not 3 <= cardinality <= 6:
                    continue
                for rule in range(int(target_rule_count[row])):
                    checked += 1
                    if int(target_cardinality[row]) != cardinality:
                        continue
                    index = cardinality - 3
                    selected = int(map_exclusion[row, rule])
                    if not 0 <= selected < 2 * cardinality + 1:
                        continue
                    path = scores[row, rule, index, : 2 * cardinality + 1]
                    stored = probabilities[row, rule, index, : 2 * cardinality + 1]
                    active_slots = tuple(range(cardinality)) + tuple(
                        range(MAX_CARDINALITY, MAX_CARDINALITY + cardinality)
                    )
                    recomputed = []
                    for excluded in range(2 * cardinality + 1):
                        value = weight * opcode_logits[row, rule, excluded]
                        for ordinal, slot in enumerate(active_slots):
                            rank = ordinal + int(ordinal >= excluded)
                            value = value + witness_logits[row, rule, slot, rank]
                        recomputed.append(value)
                    path_score_exact += int(
                        torch.allclose(
                            path,
                            torch.stack(recomputed),
                            atol=0.2,
                            rtol=2e-3,
                        )
                    )
                    recomputed_tensor = torch.stack(recomputed)
                    map_exact += int(
                        int(path.argmax()) == selected
                        and int(recomputed_tensor.argmax()) == selected
                    )
                    probability_exact += int(
                        torch.isclose(stored.sum(), torch.tensor(1.0), atol=2e-3)
                    )
                    softmax_exact += int(
                        torch.allclose(
                            stored, path.softmax(-1), atol=2e-3, rtol=2e-3
                        )
                    )
                    candidate_count = int(candidates[row, rule].ge(0).sum())
                    if candidate_count != 2 * cardinality + 1:
                        continue
                    candidate = candidates[
                        row, rule, : 2 * cardinality + 1
                    ]
                    expected_witness = torch.cat(
                        (candidate[:selected], candidate[selected + 1 :])
                    )
                    complement_exact += int(
                        int(opcode[row, rule]) == int(candidate[selected])
                        and torch.equal(
                            witness[row, rule, list(active_slots)], expected_witness
                        )
                    )
                    recomputed_marginal = torch.zeros(
                        (2 * MAX_CARDINALITY, 13)
                    )
                    for route_cardinality in range(3, 7):
                        route_index = route_cardinality - 3
                        route_slots = tuple(range(route_cardinality)) + tuple(
                            range(
                                MAX_CARDINALITY,
                                MAX_CARDINALITY + route_cardinality,
                            )
                        )
                        cardinality_mass = cardinality_probability[row, route_index]
                        if candidate_count == 2 * route_cardinality:
                            for ordinal, slot in enumerate(route_slots):
                                recomputed_marginal[slot, ordinal] += cardinality_mass
                        elif candidate_count == 2 * route_cardinality + 1:
                            route_path = probabilities[
                                row,
                                rule,
                                route_index,
                                :candidate_count,
                            ]
                            for excluded in range(candidate_count):
                                mass = cardinality_mass * route_path[excluded]
                                for ordinal, slot in enumerate(route_slots):
                                    rank = ordinal + int(ordinal >= excluded)
                                    recomputed_marginal[slot, rank] += mass
                    retained_marginal = marginal[row, rule]
                    marginal_exact += int(
                        torch.allclose(
                            retained_marginal,
                            recomputed_marginal,
                            atol=2e-3,
                            rtol=2e-3,
                        )
                    )
                    selected_ranks = recomputed_marginal[
                        list(active_slots), :candidate_count
                    ].argmax(-1)
                    full_candidate = candidates[row, rule, :candidate_count]
                    selected_positions = full_candidate[selected_ranks]
                    unique = set(map(int, selected_positions.tolist()))
                    missing = set(map(int, full_candidate.tolist())) - unique
                    independent_witness = independent["pred_witness_pointer"].long()[
                        row, rule, list(active_slots)
                    ]
                    independent_opcode = int(
                        independent["pred_rule_opcode_pointer"].long()[row, rule]
                    )
                    if len(unique) == 2 * cardinality and len(missing) == 1:
                        independent_exact += int(
                            torch.equal(independent_witness, selected_positions)
                            and independent_opcode == missing.pop()
                        )
                    else:
                        independent_exact += int(
                            bool(independent_witness.eq(-1).all())
                            and independent_opcode == -1
                        )

                    for control_name, control_raw in controls.items():
                        if control_raw is None:
                            continue
                        control = _mapping(control_raw, f"{control_name} control")
                        control_witness_logits = control[
                            "candidate_witness_logits"
                        ].float()[row, rule, :, : 2 * cardinality + 1]
                        control_opcode_logits = control[
                            "candidate_opcode_logits"
                        ].float()[row, rule, : 2 * cardinality + 1]
                        if control_name == "opcode":
                            opcode_rotation_checked += 1
                            opcode_rotation_exact += int(
                                torch.equal(
                                    control_witness_logits,
                                    witness_logits[
                                        row, rule, :, : 2 * cardinality + 1
                                    ],
                                )
                                and torch.equal(
                                    control_opcode_logits,
                                    opcode_logits[
                                        row, rule, : 2 * cardinality + 1
                                    ].roll(1),
                                )
                            )
                        else:
                            witness_rotation_checked += 1
                            witness_rotation_exact += int(
                                torch.equal(
                                    control_witness_logits,
                                    witness_logits[
                                        row, rule, :, : 2 * cardinality + 1
                                    ].roll(1, dims=-1),
                                )
                                and torch.equal(
                                    control_opcode_logits,
                                    opcode_logits[
                                        row, rule, : 2 * cardinality + 1
                                    ],
                                )
                            )
                        control_path = control["path_scores"].float()[
                            row, rule, index, : 2 * cardinality + 1
                        ]
                        control_recomputed = []
                        for excluded in range(2 * cardinality + 1):
                            value = weight * control_opcode_logits[excluded]
                            for ordinal, slot in enumerate(active_slots):
                                rank = ordinal + int(ordinal >= excluded)
                                value = value + control_witness_logits[slot, rank]
                            control_recomputed.append(value)
                        control_selected = int(
                            control["map_exclusion"].long()[row, rule]
                        )
                        control_candidate = control["candidate_positions"].long()[
                            row, rule, : 2 * cardinality + 1
                        ]
                        control_witness = control["pred_witness_pointer"].long()[
                            row, rule, list(active_slots)
                        ]
                        control_opcode = int(
                            control["rule_opcode_pointer"].long()[row, rule]
                        )
                        if 0 <= control_selected < 2 * cardinality + 1:
                            expected = torch.cat(
                                (
                                    control_candidate[:control_selected],
                                    control_candidate[control_selected + 1 :],
                                )
                            )
                            intervention_decode_exact += int(
                                torch.allclose(
                                    control_path,
                                    torch.stack(control_recomputed),
                                    atol=0.2,
                                    rtol=2e-3,
                                )
                                and int(control_path.argmax()) == control_selected
                                and control_opcode
                                == int(control_candidate[control_selected])
                                and torch.equal(control_witness, expected)
                            )
    expected_checked = checked if rows is None else sum(
        item.rule_count for item in rows
    ) * len(_mapping(evidence["arms"], "arms")) * 4
    return {
        "active_routes_checked": checked,
        "active_routes_expected": expected_checked,
        "all_route_coverage_exact": checked > 0 and checked == expected_checked,
        "all_map_argmax_exact": checked == expected_checked and map_exact == checked,
        "all_ordered_complements_exact": checked == expected_checked
        and complement_exact == checked,
        "all_probabilities_normalized": checked == expected_checked
        and probability_exact == checked,
        "all_probabilities_match_softmax": checked == expected_checked
        and softmax_exact == checked,
        "all_path_scores_recompute": checked == expected_checked
        and path_score_exact == checked,
        "all_marginal_probabilities_recompute": checked == expected_checked
        and marginal_exact == checked,
        "all_independent_decodes_recompute": checked == expected_checked
        and independent_exact == checked,
        "all_opcode_rotations_recompute": not verify_controls
        or (
            opcode_rotation_checked > 0
            and opcode_rotation_exact == opcode_rotation_checked
        ),
        "all_witness_rotations_recompute": not verify_controls
        or (
            witness_rotation_checked > 0
            and witness_rotation_exact == witness_rotation_checked
        ),
        "all_intervention_decodes_recompute": not verify_controls
        or intervention_decode_exact
        == opcode_rotation_checked + witness_rotation_checked,
    }


def _minimum(metrics: Mapping[str, object], group: str) -> float:
    return min(
        float(value[field]["rate"])
        for value in _mapping(metrics[group], group).values()
        for field in (
            "route_joint",
            "witness_pointer",
            "rule_opcode_pointer",
            "event_opcode_pointer",
        )
    )


def independent_gates(
    arms: Mapping[str, object],
    shared_initialization: bool,
    parameters: Mapping[str, int],
    development_accesses: int,
    confirmation_accesses: int,
) -> tuple[dict[str, bool], dict[str, object]]:
    def core(arm: str, mode: str) -> bool:
        value = _mapping(_mapping(arms[arm], arm)["modes"], "modes")[mode]
        return (
            all(
                float(value[view]["coherent"]["overall"][field]["rate"])
                >= THRESHOLDS["primary"]
                for view in ("canonical", "relocated")
                for field in (
                    "route_joint",
                    "state",
                    "relation_rows",
                    "witness_pointer",
                    "rule_opcode_pointer",
                    "event_opcode_pointer",
                )
            )
            and all(
                _minimum(value[view]["coherent"], group)
                >= THRESHOLDS["minimum_group"]
                for view in ("canonical", "relocated")
                for group in (
                    "by_cardinality",
                    "by_renderer",
                    "by_renderer_cardinality",
                )
            )
            and all(
                int(value[name]["route"]["exact"])
                == int(value[name]["route"]["rows"])
                == 8_000
                for name in ("alpha", "distractor")
            )
            and int(value["relocation_consistency"]["exact"])
            == int(value["relocation_consistency"]["families"])
            == 2_000
            and float(
                value["identity_deranged"]["overall"]["route_joint"]["rate"]
            )
            <= THRESHOLDS["identity_deranged_joint_max"]
        )

    def rate(arm: str, mode: str, decoder: str = "coherent") -> float:
        return float(
            arms[arm]["modes"][mode]["relocated"][decoder]["overall"][
                "route_joint"
            ]["rate"]
        )

    zero_s0 = rate("zero_update", "s0_qraw")
    zero_independent = rate("zero_update", "s0_qraw", "independent")
    zero_s1 = rate("zero_update", "s1_qraw")
    legacy_s0 = rate("legacy_uncoupled", "s0_qraw")
    legacy_s1 = rate("legacy_uncoupled", "s1_qraw")
    coupled_s0 = rate("opcode_coupled", "s0_qraw")
    coupled_s1 = rate("opcode_coupled", "s1_qraw")
    structured_s0 = rate("structured_route", "s0_qraw")
    structured_s1 = rate("structured_route", "s1_qraw")
    classifiers = {
        "coherent_decoder_repairs_independent_hardening": core(
            "zero_update", "s0_qraw"
        )
        and zero_s0 - zero_independent >= THRESHOLDS["advantage"]
        and zero_independent <= THRESHOLDS["legacy_joint_max"],
        "acute_opcode_coupling_repairs_route": core(
            "zero_update", "s1_qraw"
        )
        and zero_s1 - zero_s0 >= THRESHOLDS["advantage"]
        and zero_s0 <= THRESHOLDS["legacy_joint_max"],
        "additional_marginal_training_repairs_route": core(
            "legacy_uncoupled", "s0_qraw"
        )
        and legacy_s0 - zero_s0 >= THRESHOLDS["advantage"]
        and zero_s0 <= THRESHOLDS["legacy_joint_max"],
        "legacy_training_plus_acute_opcode_repairs_route": core(
            "legacy_uncoupled", "s1_qraw"
        )
        and legacy_s1 - max(zero_s1, legacy_s0) >= THRESHOLDS["advantage"]
        and max(zero_s1, legacy_s0) <= THRESHOLDS["legacy_joint_max"],
        "opcode_coupled_training_repairs_marginals": core(
            "opcode_coupled", "s0_qraw"
        )
        and coupled_s0 - max(legacy_s0, zero_s0) >= THRESHOLDS["advantage"]
        and max(legacy_s0, zero_s0) <= THRESHOLDS["legacy_joint_max"],
        "learned_opcode_coupling_repairs_route": core(
            "opcode_coupled", "s1_qraw"
        )
        and coupled_s1 - max(legacy_s1, coupled_s0) >= THRESHOLDS["advantage"]
        and max(legacy_s1, coupled_s0) <= THRESHOLDS["legacy_joint_max"],
        "structured_route_nll_repairs_marginals": core(
            "structured_route", "s0_qraw"
        )
        and structured_s0 - max(coupled_s0, legacy_s0) >= THRESHOLDS["advantage"]
        and max(coupled_s0, legacy_s0) <= THRESHOLDS["legacy_joint_max"],
        "structured_route_nll_repairs_route": core(
            "structured_route", "s1_qraw"
        )
        and structured_s1 - max(structured_s0, coupled_s1, legacy_s1)
        >= THRESHOLDS["advantage"]
        and max(structured_s0, coupled_s1, legacy_s1)
        <= THRESHOLDS["legacy_joint_max"],
    }
    selected = None
    if sum(map(int, classifiers.values())) == 1:
        names = {
            "coherent_decoder_repairs_independent_hardening": (
                "zero_update",
                "s0_qraw",
                "coherent_decoder",
            ),
            "acute_opcode_coupling_repairs_route": (
                "zero_update",
                "s1_qraw",
                "acute_opcode_coupling",
            ),
            "additional_marginal_training_repairs_route": (
                "legacy_uncoupled",
                "s0_qraw",
                "additional_marginal_training",
            ),
            "legacy_training_plus_acute_opcode_repairs_route": (
                "legacy_uncoupled",
                "s1_qraw",
                "legacy_training_plus_acute_opcode",
            ),
            "opcode_coupled_training_repairs_marginals": (
                "opcode_coupled",
                "s0_qraw",
                "opcode_coupled_training_marginals",
            ),
            "learned_opcode_coupling_repairs_route": (
                "opcode_coupled",
                "s1_qraw",
                "learned_opcode_coupling",
            ),
            "structured_route_nll_repairs_marginals": (
                "structured_route",
                "s0_qraw",
                "structured_route_nll_marginals",
            ),
            "structured_route_nll_repairs_route": (
                "structured_route",
                "s1_qraw",
                "structured_route_nll",
            ),
        }
        selected = names[next(name for name, value in classifiers.items() if value)]

    query_selected = None
    query_rates = None
    if selected is not None:
        qualities = {}
        for query_mode in ("qraw", "qstruct"):
            score_mode = selected[1].split("_", 1)[0]
            value = arms[selected[0]]["modes"][f"{score_mode}_{query_mode}"]
            query_rate = min(
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
            recodes = arms[selected[0]]["query_modes"][query_mode]
            counterfactual_rate = min(
                float(recodes[name]["overall"][field]["rate"])
                for name in ("target_a", "target_b")
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
                query_rate,
                cell_rate,
                counterfactual_rate,
                counterfactual_cell_rate,
            )
            qualities[query_mode] = (
                minimum >= THRESHOLDS["primary"] and invariant,
                minimum,
            )
        query_rates = {name: value[1] for name, value in qualities.items()}
        if qualities["qraw"][0]:
            query_selected = {"mode": "qraw", "mechanism": "raw_query_grounded"}
        elif (
            qualities["qstruct"][0]
            and qualities["qstruct"][1] - qualities["qraw"][1]
            >= THRESHOLDS["advantage"]
            and qualities["qraw"][1] <= THRESHOLDS["legacy_joint_max"]
        ):
            query_selected = {
                "mode": "qstruct",
                "mechanism": "structural_query_repair",
            }

    opcode_causal = True
    witness_causal = False
    if selected is not None:
        base = rate(selected[0], selected[1])
        witness = float(
            arms[selected[0]]["modes"][selected[1]]["witness_shuffled"][
                "overall"
            ]["route_joint"]["rate"]
        )
        witness_causal = (
            base - witness >= THRESHOLDS["advantage"]
            and witness <= THRESHOLDS["legacy_joint_max"]
        )
        if selected[1].startswith("s1"):
            opcode = float(
                arms[selected[0]]["modes"][selected[1]]["opcode_shuffled"][
                    "overall"
                ]["route_joint"]["rate"]
            )
            opcode_causal = (
                base - opcode >= THRESHOLDS["advantage"]
                and opcode <= THRESHOLDS["legacy_joint_max"]
            )
    gates = {
        "one_uniquely_identified_route_repair_passes": selected is not None,
        "selected_route_repair_passes_all_primary_and_cell_gates": selected
        is not None
        and core(selected[0], selected[1]),
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
        "classifiers": classifiers,
        "selected": None
        if selected is None
        else {"arm": selected[0], "mode": selected[1], "mechanism": selected[2]},
        "query_selected": query_selected,
        "query_rates": query_rates,
        "relocated_route_joint": {
            "zero_s0_independent": zero_independent,
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


def rebuild_rows(
    data_dir: Path, seed: int
) -> tuple[list[RelationTensorRow], list[RelationTensorRow], dict[str, object]]:
    board = load_board_receipt(data_dir)
    if board.get("report_sha256") != BOARD_REPORT_SHA256:
        raise ValueError("structured-route board identity differs")
    train = load_split(
        data_dir,
        board,
        filename="train.jsonl",
        split=TRAIN_SPLIT,
        expected=48_000,
    )
    _, probe, split = _split_train_families(
        train,
        derived_seed(QUALIFIED_CANARY_SEED, "dual-stream-train-probe-split"),
    )
    raw = _load_raw_train(data_dir, str(board["files"]["train.jsonl"]["sha256"]))
    relocated = _renderer_relocation_rows(
        raw, {row.family_id for row in probe}, seed=seed
    )
    return (
        [_score_train_row(row) for row in probe],
        [_score_train_row(row) for row in relocated],
        split,
    )


def verify_arm_receipts(
    checkpoint: Mapping[str, object],
    evidence: Mapping[str, object],
    report: Mapping[str, object],
) -> bool:
    checkpoint_arms = _mapping(checkpoint["arms"], "checkpoint arms")
    evidence_arms = _mapping(evidence["arms"], "evidence arms")
    report_arms = _mapping(report["arms"], "report arms")
    expected_seed = derived_seed(int(report["seed"]), "opcode-coupled-fit-order")
    frozen_digests = set()
    for name in checkpoint_arms:
        checkpoint_arm = _mapping(checkpoint_arms[name], f"checkpoint arm {name}")
        evidence_arm = _mapping(evidence_arms[name], f"evidence arm {name}")
        report_arm = _mapping(report_arms[name], f"report arm {name}")
        for key in (
            "fit_opcode_coupling_scale",
            "structured_route_objective",
            "initial_state_sha256",
            "fit",
            "compiler_trainable_state_sha256",
        ):
            if not (
                checkpoint_arm.get(key)
                == evidence_arm.get(key)
                == report_arm.get(key)
            ):
                return False
        state = _mapping(
            checkpoint_arm["compiler_trainable_state"],
            f"checkpoint trainable state {name}",
        )
        if state_dict_digest(dict(state)) != checkpoint_arm.get(
            "compiler_trainable_state_sha256"
        ):
            return False
        fit = _mapping(checkpoint_arm["fit"], f"fit receipt {name}")
        frozen_digests.add(str(fit.get("frozen_digest")))
        history = fit.get("history")
        if not isinstance(history, list):
            return False
        if name == "zero_update":
            if (
                checkpoint_arm.get("fit_opcode_coupling_scale") != 0.0
                or checkpoint_arm.get("structured_route_objective") is not False
                or fit.get("seed") is not None
                or fit.get("updates") != 0
                or fit.get("frozen_parent_unchanged") is not True
                or history
                or checkpoint_arm.get("compiler_trainable_state_sha256")
                != checkpoint_arm.get("initial_state_sha256")
            ):
                return False
        else:
            expected_weight, expected_structured = FITTED_ARMS[name]
            if (
                checkpoint_arm.get("fit_opcode_coupling_scale") != expected_weight
                or checkpoint_arm.get("structured_route_objective")
                is not expected_structured
                or fit.get("seed") != expected_seed
                or fit.get("updates") != 2_500
                or fit.get("frozen_parent_unchanged") is not True
                or len(history) != 2
                or [item.get("epoch") for item in history] != [1, 2]
                or [item.get("updates") for item in history] != [1_250, 2_500]
            ):
                return False
    return len(frozen_digests) == 1


def verify_raw_metrics(
    evidence: Mapping[str, object],
    report_arms: Mapping[str, object],
    canonical: Sequence[RelationTensorRow],
    relocated: Sequence[RelationTensorRow],
) -> bool:
    controls = {
        "alpha": [
            alpha_recode_row(row, "opcode-coupled-alpha") for row in relocated
        ],
        "distractor": [distractor_rotate_row(row) for row in relocated],
        "identity_deranged": [
            _identity_deranged_row(row, "opcode-coupled-identity")
            for row in relocated
        ],
        "opcode_shuffled": relocated,
        "witness_shuffled": relocated,
    }
    for arm_name, arm_raw in _mapping(evidence["arms"], "evidence arms").items():
        arm = _mapping(arm_raw, f"evidence arm {arm_name}")
        report_arm = _mapping(report_arms[arm_name], f"report arm {arm_name}")
        modes = _mapping(arm["modes"], "evidence modes")
        report_modes = _mapping(report_arm["modes"], "report modes")
        for mode_name, mode_raw in modes.items():
            mode = _mapping(mode_raw, f"evidence mode {mode_name}")
            report_mode = _mapping(report_modes[mode_name], "report mode")
            coherent_views = {}
            for view_name, rows in (("canonical", canonical), ("relocated", relocated)):
                view = _mapping(mode[view_name], f"evidence {view_name}")
                coherent = _mapping(view["coherent"], "coherent predictions")
                coherent_views[view_name] = coherent
                metadata = {
                    key: coherent[key]
                    for key in ("candidate_positions", "target_exclusion")
                }
                for decoder in ("coherent", "independent"):
                    metrics = score_predictions(
                        rows,
                        _mapping(view[decoder], f"{decoder} predictions"),
                        metadata,
                    )
                    if not _metrics_equal(
                        metrics,
                        _mapping(report_mode[view_name], "report view")[decoder],
                    ):
                        return False
            if relocation_consistency(
                canonical,
                relocated,
                coherent_views["canonical"],
                coherent_views["relocated"],
            ) != report_mode["relocation_consistency"]:
                return False
            relocated_view = _mapping(mode["relocated"], "relocated evidence")
            base = _mapping(relocated_view["coherent"], "base predictions")
            metadata = {
                key: base[key]
                for key in ("candidate_positions", "target_exclusion")
            }
            for name, rows in controls.items():
                raw = mode.get(name)
                if raw is None:
                    if name == "opcode_shuffled" and mode_name.startswith("s0"):
                        continue
                    return False
                raw = _mapping(raw, f"{name} predictions")
                if name in ("alpha", "distractor"):
                    if prediction_invariance(base, raw) != report_mode[name]:
                        return False
                else:
                    metrics = score_predictions(rows, raw, metadata)
                    if not _metrics_equal(
                        metrics, _mapping(report_mode[name], f"report {name}")
                    ):
                        return False
        query_evidence = _mapping(arm["query_modes"], "query evidence")
        query_report = _mapping(report_arm["query_modes"], "query report")
        for query_name in ("qraw", "qstruct"):
            query = _mapping(query_evidence[query_name], "query mode")
            base = _mapping(query["base"], "query base")
            for label in ("recode_a", "recode_b"):
                if prediction_invariance(
                    base, _mapping(query[label], "query recode")
                ) != query_report[query_name][label]:
                    return False
            for label, offset in (("target_a", 1), ("target_b", 2)):
                changed_rows = [
                    _query_target_counterfactual_row(row, offset)
                    for row in relocated
                ]
                raw = _mapping(query[label], f"query {label}")
                metadata = {
                    key: raw[key]
                    for key in ("candidate_positions", "target_exclusion")
                }
                metrics = score_predictions(changed_rows, raw, metadata)
                if not _metrics_equal(
                    metrics,
                    _mapping(query_report[query_name][label], f"report {label}"),
                ):
                    return False
    return True


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing structured-route assessment: {args.out}")

    checkpoint = _mapping(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True),
        "checkpoint",
    )
    evidence = _mapping(
        torch.load(args.evidence, map_location="cpu", weights_only=True),
        "evidence",
    )
    report = _mapping(json.loads(args.report.read_text()), "report")
    report_arms = _mapping(report["arms"], "report arms")
    shared = (
        len(
            {
                str(_mapping(value, "report arm")["initial_state_sha256"])
                for value in report_arms.values()
            }
        )
        == 1
    )
    recomputed_gates, recomputed_diagnosis = independent_gates(
        report_arms,
        shared,
        _mapping(report.get("parameters"), "report parameters"),
        int(report.get("development_accesses", -1)),
        int(report.get("confirmation_accesses", -1)),
    )
    canonical, relocated, split = rebuild_rows(
        args.data_dir, int(report["seed"])
    )
    path = verify_path_evidence(evidence, relocated)
    canonical_path = verify_path_evidence(
        evidence,
        canonical,
        view_name="canonical",
        verify_controls=False,
    )
    query_logits = verify_query_logit_evidence(evidence)
    raw_metrics_exact = verify_raw_metrics(
        evidence, report_arms, canonical, relocated
    )
    expected_arms = {"zero_update", *FITTED_ARMS}
    checks = {
        "checkpoint_schema_exact": checkpoint.get("schema") == SCHEMA,
        "evidence_schema_exact": evidence.get("schema") == EVIDENCE_SCHEMA,
        "report_schema_exact": report.get("schema") == REPORT_SCHEMA,
        "source_manifest_exact": checkpoint.get("source_manifest")
        == evidence.get("source_manifest")
        == report.get("source_manifest")
        and _source_manifest_valid(report.get("source_manifest")),
        "seed_exact": checkpoint.get("seed")
        == evidence.get("seed")
        == report.get("seed"),
        "split_rebuilt_exact": checkpoint.get("split")
        == evidence.get("split")
        == report.get("split")
        == split,
        "parameter_certificate_exact": checkpoint.get("parameters")
        == report.get("parameters")
        == EXPECTED_PARAMETERS,
        "arms_exact": set(_mapping(checkpoint["arms"], "checkpoint arms"))
        == set(_mapping(evidence["arms"], "evidence arms"))
        == set(report_arms)
        == expected_arms,
        "shared_initialization_exact": shared,
        "arm_states_and_fit_receipts_exact": verify_arm_receipts(
            checkpoint, evidence, report
        ),
        "raw_metrics_rebuilt_exact": raw_metrics_exact,
        "gates_independently_recompute_exact": recomputed_gates
        == report.get("gates"),
        "diagnosis_independently_recomputes_exact": recomputed_diagnosis
        == report.get("diagnosis"),
        "decision_recomputes_exact": report.get("decision")
        == (
            "authorize_new_fresh_board_source"
            if all(recomputed_gates.values())
            else "reject_opcode_coupled_before_fresh_board"
        ),
        "path_map_exact": path["all_map_argmax_exact"],
        "path_route_coverage_exact": path["all_route_coverage_exact"],
        "canonical_path_route_coverage_exact": canonical_path[
            "all_route_coverage_exact"
        ],
        "canonical_path_map_exact": canonical_path["all_map_argmax_exact"],
        "canonical_path_ordered_complement_exact": canonical_path[
            "all_ordered_complements_exact"
        ],
        "canonical_path_probability_exact": canonical_path[
            "all_probabilities_normalized"
        ],
        "canonical_path_probability_softmax_exact": canonical_path[
            "all_probabilities_match_softmax"
        ],
        "canonical_path_scores_recompute_exact": canonical_path[
            "all_path_scores_recompute"
        ],
        "canonical_marginal_probabilities_recompute_exact": canonical_path[
            "all_marginal_probabilities_recompute"
        ],
        "canonical_independent_decoder_recomputes_exact": canonical_path[
            "all_independent_decodes_recompute"
        ],
        "path_ordered_complement_exact": path["all_ordered_complements_exact"],
        "path_probability_exact": path["all_probabilities_normalized"],
        "path_probability_softmax_exact": path[
            "all_probabilities_match_softmax"
        ],
        "path_scores_recompute_exact": path["all_path_scores_recompute"],
        "marginal_probabilities_recompute_exact": path[
            "all_marginal_probabilities_recompute"
        ],
        "independent_decoder_recomputes_exact": path[
            "all_independent_decodes_recompute"
        ],
        "opcode_rotation_recomputes_exact": path[
            "all_opcode_rotations_recompute"
        ],
        "witness_rotation_recomputes_exact": path[
            "all_witness_rotations_recompute"
        ],
        "intervention_decodes_recompute_exact": path[
            "all_intervention_decodes_recompute"
        ],
        "query_semantic_logits_recompute_exact": query_logits[
            "all_query_semantic_argmax_exact"
        ],
        "query_pointer_logits_recompute_exact": query_logits[
            "all_query_pointer_argmax_exact"
        ],
        "query_logits_finite": query_logits["all_query_logits_finite"],
        "zero_scored_reads": int(report.get("development_accesses", -1)) == 0
        and int(report.get("confirmation_accesses", -1)) == 0
        and int(evidence.get("development_accesses", -1)) == 0
        and int(evidence.get("confirmation_accesses", -1)) == 0
        and int(checkpoint.get("development_accesses", -1)) == 0
        and int(checkpoint.get("confirmation_accesses", -1)) == 0,
    }
    all_checks = all(checks.values())
    assessment = {
        "schema": ASSESSMENT_SCHEMA,
        "files": {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "evidence_sha256": sha256_file(args.evidence),
            "report_sha256": sha256_file(args.report),
        },
        "checks": checks,
        "path_evidence": path,
        "canonical_path_evidence": canonical_path,
        "query_logit_evidence": query_logits,
        "all_checks_pass": all_checks,
        "decision": report.get("decision") if all_checks else "reject_assessment",
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    atomic_json_save(assessment, args.out)
    print(json.dumps(assessment, sort_keys=True))
    if not all_checks:
        raise SystemExit("structured-route independent assessment failed")


def main() -> None:
    try:
        _main()
    except SystemExit:
        raise
    except Exception as exc:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--out", type=Path)
        parsed, _ = parser.parse_known_args()
        failure = {
            "schema": ASSESSMENT_SCHEMA,
            "checks": {"assessment_execution_completed": False},
            "all_checks_pass": False,
            "decision": "reject_assessment",
            "error_type": type(exc).__name__,
            "development_accesses": 0,
            "confirmation_accesses": 0,
        }
        if parsed.out is not None and not parsed.out.exists():
            atomic_json_save(failure, parsed.out)
        print(json.dumps(failure, sort_keys=True))
        raise SystemExit("structured-route independent assessment aborted") from None


if __name__ == "__main__":
    main()
