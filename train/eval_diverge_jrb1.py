#!/usr/bin/env python3
"""Evaluate one frozen DIVERGE-JRB1 joint-register gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_eal1_data import TRANSFER_DEPTHS, canonical_sha256, scan_integer_spans
from diverge_eal1_runtime import (
    EpisodeLawPacket,
    compile_episode_laws as compile_complete_roles,
    execute_program,
    module_state_sha256,
    sha256_path,
)
from diverge_eal2_runtime import hard_temporal_assignment, load_reader
from diverge_jrb1_data import DEVELOPMENT_EPISODES, validate_evaluation_episode
from diverge_jrb1_runtime import (
    load_binder,
    tensorize_register_sources,
    tensorize_temporal_without_register_scan,
)
from diverge_ncp1_runtime import greedy_ctc_decode, load_pointer, tensorize_commands
from eval_diverge_eal1 import _load_jsonl
from eval_diverge_ncp1 import _program_score


SCHEMA = "shohin-diverge-jrb1-evaluation-v1"


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_report(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"JRB1 report hash differs: {path}")
    return json.loads(path.read_text())


def _source_records(
    public: Sequence[Mapping[str, Any]],
    *,
    group: str,
    text_key: str,
    registers_key: str,
) -> list[dict[str, Any]]:
    return [
        {"source_text": item[text_key], "registers": episode[registers_key]}
        for episode in public
        for item in episode[group]
    ]


@torch.no_grad()
def _predict_mentions(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    mention_count: int,
    batch_size: int,
    rotate_table: bool = False,
    canonicalize_rotation: bool = False,
) -> list[tuple[int, ...]]:
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        tensors = tensorize_register_sources(
            batch,
            device,
            text_key="source_text",
            mention_count=mention_count,
            rotate_register_table=rotate_table,
        )
        if tensors[4] is None or tensors[5] is None:
            raise RuntimeError("JRB1 mention tensorization omitted bounds")
        predictions = model.forward_mentions(  # type: ignore[attr-defined]
            *tensors[:4], tensors[4], tensors[5]
        ).argmax(dim=-1)
        for row in predictions.detach().cpu():
            values = tuple(int(value) for value in row)
            if canonicalize_rotation:
                values = tuple(1 - value for value in values)
            output.append(values)
    return output


@torch.no_grad()
def _predict_queries(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    rotate_table: bool = False,
    canonicalize_rotation: bool = False,
) -> list[int]:
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        tensors = tensorize_register_sources(
            batch,
            device,
            text_key="source_text",
            mention_count=None,
            rotate_register_table=rotate_table,
        )
        predictions = model.forward_query(*tensors[:4]).argmax(dim=-1)  # type: ignore[attr-defined]
        for value in predictions.detach().cpu():
            prediction = int(value)
            output.append(1 - prediction if canonicalize_rotation else prediction)
    return output


@torch.no_grad()
def _predict_temporal(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    register_predictions: Sequence[Sequence[int]],
    *,
    device: torch.device,
    batch_size: int,
) -> list[tuple[int, ...] | None]:
    if len(records) != len(register_predictions):
        raise RuntimeError("JRB1 temporal/register count differs")
    output: list[tuple[int, ...] | None] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        byte_ids, attention, bounds = tensorize_temporal_without_register_scan(
            batch, device, text_key="source_text"
        )
        logits = model(byte_ids, attention, bounds)
        for index, register_values in enumerate(
            register_predictions[start : start + len(batch)]
        ):
            registers = tuple(int(value) for value in register_values)
            if registers.count(0) != 2 or registers.count(1) != 2:
                output.append(None)
            else:
                output.append(hard_temporal_assignment(logits[index], registers))
    return output


@torch.no_grad()
def _predict_programs(
    model: torch.nn.Module,
    public: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> list[tuple[int, ...]]:
    records = [
        {"source_text": item["command_text"], "aliases": episode["aliases"]}
        for episode in public
        for item in episode["transfer"]
    ]
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        command_ids, command_mask, alias_ids, alias_mask, lengths = tensorize_commands(
            batch, device
        )
        output.extend(
            greedy_ctc_decode(
                model(command_ids, command_mask, alias_ids, alias_mask), lengths
            )
        )
    return output


def _sequence_score(
    predicted: Sequence[Sequence[int]], gold: Sequence[Sequence[int]]
) -> dict[str, Any]:
    if len(predicted) != len(gold):
        raise RuntimeError("JRB1 sequence score count differs")
    exact = elements = correct = 0
    for guess, target in zip(predicted, gold, strict=True):
        guess_tuple = tuple(int(value) for value in guess)
        target_tuple = tuple(int(value) for value in target)
        exact += guess_tuple == target_tuple
        correct += sum(
            left == right for left, right in zip(guess_tuple, target_tuple)
        )
        elements += len(target_tuple)
    return {
        "exact": exact,
        "total": len(gold),
        "exact_rate": exact / max(1, len(gold)),
        "element_exact": correct,
        "elements": elements,
        "element_rate": correct / max(1, elements),
    }


def _scalar_score(predicted: Sequence[int], gold: Sequence[int]) -> dict[str, Any]:
    if len(predicted) != len(gold):
        raise RuntimeError("JRB1 scalar score count differs")
    exact = sum(
        int(left) == int(right) for left, right in zip(predicted, gold, strict=True)
    )
    return {"exact": exact, "total": len(gold), "rate": exact / max(1, len(gold))}


def _temporal_score(
    predicted: Sequence[Sequence[int] | None], gold: Sequence[Sequence[int]]
) -> dict[str, Any]:
    normalized = [() if value is None else tuple(value) for value in predicted]
    report = _sequence_score(normalized, gold)
    report["invalid_register_group"] = sum(value is None for value in predicted)
    return report


def _compose_roles(
    temporal: Sequence[Sequence[int] | None],
    registers: Sequence[Sequence[int]],
) -> list[tuple[int, ...] | None]:
    if len(temporal) != len(registers):
        raise RuntimeError("JRB1 role-source count differs")
    output = []
    for time_values, register_values in zip(temporal, registers, strict=True):
        if time_values is None:
            output.append(None)
            continue
        roles = tuple(
            int(time_value) * 2 + int(register_value)
            for time_value, register_value in zip(
                time_values, register_values, strict=True
            )
        )
        output.append(roles if sorted(roles) == [0, 1, 2, 3] else None)
    return output


def _compile_packets(
    public: Sequence[Mapping[str, Any]],
    roles: Sequence[Sequence[int] | None],
    *,
    text_key: str,
    hash_key: str,
    owner_state_sha256: str,
) -> list[EpisodeLawPacket | None]:
    stride = len(public[0]["evidence"])
    if len(roles) != len(public) * stride:
        raise RuntimeError("JRB1 complete-role count differs")
    output = []
    for episode_index, episode in enumerate(public):
        assignments = roles[episode_index * stride : (episode_index + 1) * stride]
        if any(value is None for value in assignments):
            output.append(None)
            continue
        visible = {
            "aliases": episode["aliases"],
            "evidence": [
                {
                    **item,
                    "source_text": item[text_key],
                    "source_sha256": item[hash_key],
                }
                for item in episode["evidence"]
            ],
        }
        output.append(
            compile_complete_roles(
                visible,
                assignments,  # type: ignore[arg-type]
                reader_state_sha256=owner_state_sha256,
            ).packet
        )
    return output


def _decode_initial_states(
    public: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[int]],
    *,
    text_key: str,
) -> list[tuple[int, int] | None]:
    transfers = [item for episode in public for item in episode["transfer"]]
    if len(transfers) != len(predictions):
        raise RuntimeError("JRB1 initial-state count differs")
    output = []
    for item, predicted in zip(transfers, predictions, strict=True):
        text = str(item[text_key])
        values = tuple(int(text[start:end]) for start, end in scan_integer_spans(text))
        registers = tuple(int(value) for value in predicted)
        if len(values) != 2 or sorted(registers) != [0, 1]:
            output.append(None)
            continue
        state = [0, 0]
        for value, register in zip(values, registers, strict=True):
            state[register] = value
        output.append((state[0], state[1]))
    return output


def _state_binding_score(
    decoded: Sequence[Sequence[int] | None],
    assessor: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gold = [
        tuple(int(value) for value in target["state"])
        for episode in assessor
        for target in episode["initial_targets"]
    ]
    normalized = [() if value is None else tuple(value) for value in decoded]
    report = _sequence_score(normalized, gold)
    report["invalid"] = sum(value is None for value in decoded)
    return report


def _execution_score(
    packets: Sequence[EpisodeLawPacket | None],
    public: Sequence[Mapping[str, Any]],
    assessor: Sequence[Mapping[str, Any]],
    program_predictions: Sequence[Sequence[int]],
    initial_states: Sequence[Sequence[int] | None],
    query_predictions: Sequence[int],
) -> dict[str, Any]:
    program_cursor = query_cursor = 0
    state_exact = answer_exact = programs = queries = abstained = 0
    by_depth: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for packet, visible, hidden in zip(packets, public, assessor, strict=True):
        terminal_by_id = {
            str(item["program_id"]): tuple(int(value) for value in item["terminal_state"])
            for item in hidden["transfer"]
        }
        predicted_by_id: dict[str, tuple[int, int] | None] = {}
        for transfer in visible["transfer"]:
            depth = int(transfer["depth"])
            symbols = tuple(int(value) for value in program_predictions[program_cursor])
            initial = initial_states[program_cursor]
            prediction = None
            if (
                packet is not None
                and initial is not None
                and len(symbols) == depth
                and all(0 <= value < len(visible["aliases"]) for value in symbols)
            ):
                prediction = execute_program(
                    packet,
                    {
                        "depth": depth,
                        "initial_state": list(initial),
                        "symbols": [visible["aliases"][value] for value in symbols],
                    },
                )
            program_id = str(transfer["program_id"])
            predicted_by_id[program_id] = prediction
            gold = terminal_by_id[program_id]
            exact = prediction == gold
            state_exact += exact
            programs += 1
            abstained += prediction is None
            by_depth[depth][0] += int(exact)
            by_depth[depth][1] += 1
            program_cursor += 1
        for query, hidden_query in zip(
            visible["queries"], hidden["query_targets"], strict=True
        ):
            program_id = str(query["program_id"])
            if program_id != str(hidden_query["program_id"]):
                raise RuntimeError("JRB1 query program identity differs")
            predicted_state = predicted_by_id[program_id]
            predicted_register = int(query_predictions[query_cursor])
            gold_register = int(hidden_query["register_index"])
            gold_answer = terminal_by_id[program_id][gold_register]
            answer_exact += (
                predicted_state is not None
                and predicted_register in (0, 1)
                and predicted_state[predicted_register] == gold_answer
            )
            queries += 1
            query_cursor += 1
    if program_cursor != len(program_predictions) or query_cursor != len(
        query_predictions
    ):
        raise RuntimeError("JRB1 execution cursor differs")
    if tuple(sorted(by_depth)) != tuple(sorted(TRANSFER_DEPTHS)):
        raise RuntimeError("JRB1 transfer depth board differs")
    depth_report = {
        str(depth): {"exact": value[0], "total": value[1], "rate": value[0] / value[1]}
        for depth, value in sorted(by_depth.items())
    }
    return {
        "state_exact": state_exact,
        "programs": programs,
        "state_exact_rate": state_exact / max(1, programs),
        "answer_exact": answer_exact,
        "queries": queries,
        "answer_exact_rate": answer_exact / max(1, queries),
        "abstained_programs": abstained,
        "by_depth": depth_report,
        "minimum_depth_rate": min(value["rate"] for value in depth_report.values()),
    }


def _arm_predictions(
    binder: torch.nn.Module,
    public: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    evidence_text_key: str,
    initial_text_key: str,
    query_text_key: str,
    registers_key: str,
    rotate_table: bool = False,
    canonicalize_rotation: bool = False,
) -> dict[str, Any]:
    evidence_records = _source_records(
        public,
        group="evidence",
        text_key=evidence_text_key,
        registers_key=registers_key,
    )
    initial_records = _source_records(
        public,
        group="transfer",
        text_key=initial_text_key,
        registers_key=registers_key,
    )
    query_records = _source_records(
        public,
        group="queries",
        text_key=query_text_key,
        registers_key=registers_key,
    )
    return {
        "evidence_records": evidence_records,
        "evidence": _predict_mentions(
            binder,
            evidence_records,
            device=device,
            mention_count=4,
            batch_size=batch_size,
            rotate_table=rotate_table,
            canonicalize_rotation=canonicalize_rotation,
        ),
        "initial": _predict_mentions(
            binder,
            initial_records,
            device=device,
            mention_count=2,
            batch_size=batch_size,
            rotate_table=rotate_table,
            canonicalize_rotation=canonicalize_rotation,
        ),
        "query": _predict_queries(
            binder,
            query_records,
            device=device,
            batch_size=batch_size,
            rotate_table=rotate_table,
            canonicalize_rotation=canonicalize_rotation,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eal2-checkpoint", type=Path, required=True)
    parser.add_argument("--eal2-checkpoint-sha256", required=True)
    parser.add_argument("--eal2-development-report", type=Path, required=True)
    parser.add_argument("--eal2-development-report-sha256", required=True)
    parser.add_argument("--ncp1-checkpoint", type=Path, required=True)
    parser.add_argument("--ncp1-checkpoint-sha256", required=True)
    parser.add_argument("--ncp1-development-report", type=Path, required=True)
    parser.add_argument("--ncp1-development-report-sha256", required=True)
    parser.add_argument("--treatment-checkpoint", type=Path, required=True)
    parser.add_argument("--treatment-checkpoint-sha256", required=True)
    parser.add_argument("--treatment-report", type=Path, required=True)
    parser.add_argument("--treatment-report-sha256", required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint-sha256", required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--control-report-sha256", required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--public-data-sha256", required=True)
    parser.add_argument("--assessor-data", type=Path, required=True)
    parser.add_argument("--assessor-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing JRB1 evaluation")

    eal2_parent = _load_report(
        args.eal2_development_report, args.eal2_development_report_sha256
    )
    ncp1_parent = _load_report(
        args.ncp1_development_report, args.ncp1_development_report_sha256
    )
    if (
        eal2_parent.get("status") != "pass"
        or eal2_parent.get("checkpoint_sha256") != args.eal2_checkpoint_sha256
        or ncp1_parent.get("status") != "pass"
        or ncp1_parent["parent_eal2"]["checkpoint_sha256"]
        != args.eal2_checkpoint_sha256
        or ncp1_parent["training"]["treatment_checkpoint_sha256"]
        != args.ncp1_checkpoint_sha256
    ):
        raise SystemExit("JRB1 qualified parent custody differs")
    treatment_report = _load_report(
        args.treatment_report, args.treatment_report_sha256
    )
    control_report = _load_report(args.control_report, args.control_report_sha256)
    if (
        treatment_report.get("status") != "complete"
        or treatment_report.get("arm") != "treatment"
        or control_report.get("status") != "complete"
        or control_report.get("arm") != "shuffled_table"
    ):
        raise SystemExit("JRB1 training report custody differs")
    public = _load_jsonl(args.public_data, args.public_data_sha256)
    assessor = _load_jsonl(args.assessor_data, args.assessor_data_sha256)
    if len(public) != DEVELOPMENT_EPISODES or len(assessor) != DEVELOPMENT_EPISODES:
        raise SystemExit("JRB1 evaluation board geometry differs")
    for visible, hidden in zip(public, assessor, strict=True):
        validate_evaluation_episode(visible, hidden)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader, _ = load_reader(args.eal2_checkpoint, args.eal2_checkpoint_sha256)
    pointer, _ = load_pointer(args.ncp1_checkpoint, args.ncp1_checkpoint_sha256)
    treatment, treatment_checkpoint = load_binder(
        args.treatment_checkpoint, args.treatment_checkpoint_sha256
    )
    control, control_checkpoint = load_binder(
        args.control_checkpoint, args.control_checkpoint_sha256
    )
    reader = reader.to(device).eval()
    pointer = pointer.to(device).eval()
    treatment = treatment.to(device).eval()
    control = control.to(device).eval()
    reader_hash = module_state_sha256(reader)
    pointer_hash = module_state_sha256(pointer)
    treatment_hash = module_state_sha256(treatment)
    control_hash = module_state_sha256(control)

    gold_evidence = [
        tuple(int(value) % 2 for value in item["numeric_role_ids"])
        for episode in assessor
        for item in episode["evidence"]
    ]
    gold_temporal = [
        tuple(int(value) // 2 for value in item["numeric_role_ids"])
        for episode in assessor
        for item in episode["evidence"]
    ]
    gold_roles = [
        tuple(int(value) for value in item["numeric_role_ids"])
        for episode in assessor
        for item in episode["evidence"]
    ]
    gold_initial_mentions = [
        tuple(int(value) for value in item["mention_register_targets"])
        for episode in assessor
        for item in episode["initial_targets"]
    ]
    gold_queries = [
        int(item["register_index"])
        for episode in assessor
        for item in episode["query_targets"]
    ]
    gold_programs = [
        tuple(int(value) for value in item["targets"])
        for episode in assessor
        for item in episode["command_targets"]
    ]
    program_predictions = _predict_programs(
        pointer, public, device=device, batch_size=args.batch_size
    )
    program_report = _program_score(program_predictions, gold_programs)

    arm_specs = {
        "treatment": {
            "model": treatment,
            "evidence_text_key": "source_text",
            "initial_text_key": "initial_text",
            "query_text_key": "query_text",
            "registers_key": "registers",
            "hash_key": "source_sha256",
        },
        "renamed": {
            "model": treatment,
            "evidence_text_key": "renamed_source_text",
            "initial_text_key": "renamed_initial_text",
            "query_text_key": "renamed_query_text",
            "registers_key": "renamed_registers",
            "hash_key": "renamed_source_sha256",
        },
        "table_permutation": {
            "model": treatment,
            "evidence_text_key": "source_text",
            "initial_text_key": "initial_text",
            "query_text_key": "query_text",
            "registers_key": "registers",
            "hash_key": "source_sha256",
            "rotate_table": True,
            "canonicalize_rotation": True,
        },
        "source_scrub": {
            "model": treatment,
            "evidence_text_key": "register_scrubbed_text",
            "initial_text_key": "register_scrubbed_initial_text",
            "query_text_key": "register_scrubbed_query_text",
            "registers_key": "registers",
            "hash_key": "register_scrubbed_sha256",
        },
        "shuffled_table_model": {
            "model": control,
            "evidence_text_key": "source_text",
            "initial_text_key": "initial_text",
            "query_text_key": "query_text",
            "registers_key": "registers",
            "hash_key": "source_sha256",
        },
    }
    arm_reports = {}
    for name, spec in arm_specs.items():
        predictions = _arm_predictions(
            spec["model"],
            public,
            device=device,
            batch_size=args.batch_size,
            evidence_text_key=spec["evidence_text_key"],
            initial_text_key=spec["initial_text_key"],
            query_text_key=spec["query_text_key"],
            registers_key=spec["registers_key"],
            rotate_table=bool(spec.get("rotate_table", False)),
            canonicalize_rotation=bool(spec.get("canonicalize_rotation", False)),
        )
        temporal = _predict_temporal(
            reader,
            predictions["evidence_records"],
            predictions["evidence"],
            device=device,
            batch_size=args.batch_size,
        )
        roles = _compose_roles(temporal, predictions["evidence"])
        owner_hash = canonical_sha256(
            {"reader": reader_hash, "binder": treatment_hash if name != "shuffled_table_model" else control_hash}
        )
        packets = _compile_packets(
            public,
            roles,
            text_key=spec["evidence_text_key"],
            hash_key=spec["hash_key"],
            owner_state_sha256=owner_hash,
        )
        initial_states = _decode_initial_states(
            public, predictions["initial"], text_key=spec["initial_text_key"]
        )
        arm_reports[name] = {
            "evidence_register": _sequence_score(
                predictions["evidence"], gold_evidence
            ),
            "temporal": _temporal_score(temporal, gold_temporal),
            "complete_roles": _sequence_score(
                [() if value is None else value for value in roles], gold_roles
            ),
            "law_commits": sum(packet is not None for packet in packets),
            "law_episodes": len(packets),
            "initial_register": _sequence_score(
                predictions["initial"], gold_initial_mentions
            ),
            "initial_state": _state_binding_score(initial_states, assessor),
            "query_register": _scalar_score(predictions["query"], gold_queries),
            "execution": _execution_score(
                packets,
                public,
                assessor,
                program_predictions,
                initial_states,
                predictions["query"],
            ),
        }

    runtime_path = Path(__file__).with_name("diverge_jrb1_runtime.py")
    runtime_source = runtime_path.read_text()
    source_audit = {
        "runtime": str(runtime_path),
        "runtime_sha256": hashlib.sha256(runtime_source.encode()).hexdigest(),
        "contains_exact_register_search": any(
            value in runtime_source
            for value in ("scan_register_ids", "re.finditer", "re.search", ".find(")
        ),
        "uses_dynamic_register_pointer": "torch.einsum" in runtime_source,
        "uses_exact_numeric_span_carrier": "scan_integer_spans" in runtime_source,
        "typed_initial_absent": all(
            "initial_state" not in item
            for episode in public
            for item in episode["transfer"]
        ),
        "typed_query_absent": all(
            "register_index" not in item
            for episode in public
            for item in episode["queries"]
        ),
    }
    treatment_arm = arm_reports["treatment"]
    renamed_arm = arm_reports["renamed"]
    permutation_arm = arm_reports["table_permutation"]
    scrub_arm = arm_reports["source_scrub"]
    control_arm = arm_reports["shuffled_table_model"]
    conditions = {
        "parent_eal2_and_ncp1_pass": True,
        "command_program_at_least_99_percent": program_report["exact_rate"] >= 0.99,
        "command_depth_floor_at_least_95_percent": program_report[
            "minimum_depth_rate"
        ]
        >= 0.95,
        "treatment_evidence_register_at_least_99_percent": treatment_arm[
            "evidence_register"
        ]["exact_rate"]
        >= 0.99,
        "treatment_initial_register_at_least_99_percent": treatment_arm[
            "initial_register"
        ]["exact_rate"]
        >= 0.99,
        "treatment_query_register_at_least_99_percent": treatment_arm[
            "query_register"
        ]["rate"]
        >= 0.99,
        "treatment_complete_roles_at_least_99_percent": treatment_arm[
            "complete_roles"
        ]["exact_rate"]
        >= 0.99,
        "treatment_law_commits_at_least_99_percent": treatment_arm["law_commits"]
        / treatment_arm["law_episodes"]
        >= 0.99,
        "treatment_initial_state_at_least_99_percent": treatment_arm["initial_state"][
            "exact_rate"
        ]
        >= 0.99,
        "treatment_terminal_state_at_least_99_percent": treatment_arm["execution"][
            "state_exact_rate"
        ]
        >= 0.99,
        "treatment_answer_at_least_99_percent": treatment_arm["execution"][
            "answer_exact_rate"
        ]
        >= 0.99,
        "treatment_depth_floor_at_least_95_percent": treatment_arm["execution"][
            "minimum_depth_rate"
        ]
        >= 0.95,
        "renamed_register_state_answer_at_least_99_percent": min(
            renamed_arm["evidence_register"]["exact_rate"],
            renamed_arm["initial_register"]["exact_rate"],
            renamed_arm["query_register"]["rate"],
            renamed_arm["execution"]["state_exact_rate"],
            renamed_arm["execution"]["answer_exact_rate"],
        )
        >= 0.99,
        "table_permutation_equivariance_at_least_99_percent": min(
            permutation_arm["evidence_register"]["exact_rate"],
            permutation_arm["initial_register"]["exact_rate"],
            permutation_arm["query_register"]["rate"],
            permutation_arm["execution"]["state_exact_rate"],
            permutation_arm["execution"]["answer_exact_rate"],
        )
        >= 0.99,
        "register_source_scrub_state_and_answer_at_most_5_percent": max(
            scrub_arm["execution"]["state_exact_rate"],
            scrub_arm["execution"]["answer_exact_rate"],
        )
        <= 0.05,
        "shuffled_table_model_state_and_answer_at_most_5_percent": max(
            control_arm["execution"]["state_exact_rate"],
            control_arm["execution"]["answer_exact_rate"],
        )
        <= 0.05,
        "matched_initialization_data_and_schedule": treatment_report[
            "initial_state_sha256"
        ]
        == control_report["initial_state_sha256"]
        and treatment_report["data_sha256"] == control_report["data_sha256"]
        and treatment_report["updates"] == control_report["updates"]
        and treatment_report["batch_size"] == control_report["batch_size"]
        and treatment_report["learning_rate"] == control_report["learning_rate"],
        "checkpoint_report_custody_matches": treatment_checkpoint[
            "model_state_sha256"
        ]
        == treatment_report["final_state_sha256"]
        and control_checkpoint["model_state_sha256"]
        == control_report["final_state_sha256"],
        "qualified_parent_weights_bit_identical": reader_hash
        == eal2_parent["reader_state_sha256"]
        and args.ncp1_checkpoint_sha256
        == ncp1_parent["training"]["treatment_checkpoint_sha256"],
        "runtime_has_no_exact_register_search": not source_audit[
            "contains_exact_register_search"
        ],
        "typed_initial_and_query_deleted": source_audit["typed_initial_absent"]
        and source_audit["typed_query_absent"],
    }
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "source_commit": args.source_commit,
        "parents": {
            "eal2_checkpoint_sha256": args.eal2_checkpoint_sha256,
            "eal2_development_report_sha256": args.eal2_development_report_sha256,
            "ncp1_checkpoint_sha256": args.ncp1_checkpoint_sha256,
            "ncp1_development_report_sha256": args.ncp1_development_report_sha256,
            "reader_state_sha256": reader_hash,
            "pointer_state_sha256": pointer_hash,
        },
        "training": {
            "treatment_checkpoint_sha256": args.treatment_checkpoint_sha256,
            "treatment_report_sha256": args.treatment_report_sha256,
            "control_checkpoint_sha256": args.control_checkpoint_sha256,
            "control_report_sha256": args.control_report_sha256,
            "treatment_state_sha256": treatment_hash,
            "control_state_sha256": control_hash,
        },
        "data": {
            "public": str(args.public_data),
            "public_sha256": args.public_data_sha256,
            "assessor": str(args.assessor_data),
            "assessor_sha256": args.assessor_data_sha256,
        },
        "program": program_report,
        "arms": arm_reports,
        "source_audit": source_audit,
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "status": report["status"],
                "state_rate": treatment_arm["execution"]["state_exact_rate"],
                "answer_rate": treatment_arm["execution"]["answer_exact_rate"],
                "renamed_state_rate": renamed_arm["execution"]["state_exact_rate"],
                "scrub_state_rate": scrub_arm["execution"]["state_exact_rate"],
                "control_state_rate": control_arm["execution"]["state_exact_rate"],
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
