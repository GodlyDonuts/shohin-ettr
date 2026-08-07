#!/usr/bin/env python3
"""Evaluate one frozen DIVERGE-OQB1 occurrence-quotient register gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_oqb1_data import (
    REPORT_SCHEMA as DATA_REPORT_SCHEMA,
    DEVELOPMENT_EPISODES,
    validate_evaluation_episode,
)
from diverge_eal1_data import canonical_sha256
from diverge_oqb1_runtime import (
    QuotientMode,
    load_binder,
    tensorize_quotient_sources,
)
from diverge_eal1_data import TRANSFER_DEPTHS
from diverge_eal1_runtime import (
    EpisodeLawPacket,
    execute_program,
    module_state_sha256,
    sha256_path,
)
from diverge_eal2_runtime import load_reader
from diverge_ncp1_runtime import load_pointer
from eval_diverge_eal1 import _load_jsonl
from eval_diverge_jrb1 import (
    _compile_packets,
    _compose_roles,
    _decode_initial_states,
    _predict_programs,
    _predict_temporal,
    _program_score,
    _scalar_score,
    _sequence_score,
)


SCHEMA = "shohin-diverge-oqb1-evaluation-v1"


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
        raise RuntimeError(f"OQB1 report hash differs: {path}")
    return json.loads(path.read_text())


def _table(values: Sequence[str], reverse: bool) -> tuple[str, str]:
    table = tuple(str(value) for value in values)
    if len(table) != 2 or len(set(table)) != 2:
        raise RuntimeError("OQB1 evaluator table differs")
    return (table[1], table[0]) if reverse else (table[0], table[1])


def _records(
    public: Sequence[Mapping[str, Any]],
    *,
    group: str,
    text_key: str,
    table_key: str,
    reverse_table: bool,
) -> list[dict[str, Any]]:
    output = []
    for episode in public:
        for index, item in enumerate(episode[group]):
            output.append(
                {
                    "source_text": item[text_key],
                    "registers": list(_table(episode[table_key], reverse_table)),
                    "serial": f"{episode['episode_id']}|{group}|{index}",
                }
            )
    return output


@torch.no_grad()
def _predict_mentions(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    mention_count: int,
    batch_size: int,
    mode: QuotientMode,
) -> list[tuple[int, ...]]:
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        tensors = tensorize_quotient_sources(
            batch,
            device,
            text_key="source_text",
            mention_count=mention_count,
            mode=mode,
        )
        if tensors[2] is None or tensors[3] is None:
            raise RuntimeError("OQB1 mention tensorization omitted bounds")
        predictions = model.forward_mentions(  # type: ignore[attr-defined]
            tensors[0], tensors[1], tensors[2], tensors[3]
        ).argmax(dim=-1)
        for row, valid in zip(
            predictions.detach().cpu(), tensors[4].cpu(), strict=True
        ):
            output.append(tuple(int(value) for value in row) if bool(valid) else ())
    return output


@torch.no_grad()
def _predict_queries(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    mode: QuotientMode,
) -> list[int]:
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        tensors = tensorize_quotient_sources(
            batch,
            device,
            text_key="source_text",
            mention_count=None,
            mode=mode,
        )
        predictions = model.forward_query(tensors[0], tensors[1]).argmax(dim=-1)  # type: ignore[attr-defined]
        for value, valid in zip(
            predictions.detach().cpu(), tensors[4].cpu(), strict=True
        ):
            output.append(int(value) if bool(valid) else -1)
    return output


def _predict_arm(
    model: torch.nn.Module,
    public: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    evidence_text_key: str,
    initial_text_key: str,
    query_text_key: str,
    table_key: str,
    evidence_reverse: bool = False,
    initial_reverse: bool = False,
    query_reverse: bool = False,
    mode: QuotientMode = "coherent",
) -> dict[str, Any]:
    evidence_records = _records(
        public,
        group="evidence",
        text_key=evidence_text_key,
        table_key=table_key,
        reverse_table=evidence_reverse,
    )
    initial_records = _records(
        public,
        group="transfer",
        text_key=initial_text_key,
        table_key=table_key,
        reverse_table=initial_reverse,
    )
    query_records = _records(
        public,
        group="queries",
        text_key=query_text_key,
        table_key=table_key,
        reverse_table=query_reverse,
    )
    return {
        "evidence_records": evidence_records,
        "evidence": _predict_mentions(
            model,
            evidence_records,
            device=device,
            mention_count=4,
            batch_size=batch_size,
            mode=mode,
        ),
        "initial": _predict_mentions(
            model,
            initial_records,
            device=device,
            mention_count=2,
            batch_size=batch_size,
            mode=mode,
        ),
        "query": _predict_queries(
            model,
            query_records,
            device=device,
            batch_size=batch_size,
            mode=mode,
        ),
    }


def _position_targets(
    public: Sequence[Mapping[str, Any]],
    assessor: Sequence[Mapping[str, Any]],
    *,
    group: str,
    table_key: str,
    canonical_key: str,
    reverse_table: bool,
) -> list[tuple[int, ...]] | list[int]:
    output: list[Any] = []
    for visible, hidden in zip(public, assessor, strict=True):
        table = _table(visible[table_key], reverse_table)
        canonical = tuple(str(value) for value in hidden[canonical_key])
        canonical_to_position = tuple(table.index(value) for value in canonical)
        if group == "evidence":
            output.extend(
                tuple(
                    canonical_to_position[int(value) % 2]
                    for value in item["numeric_role_ids"]
                )
                for item in hidden["evidence"]
            )
        elif group == "initial":
            output.extend(
                tuple(
                    canonical_to_position[int(value)]
                    for value in item["mention_register_targets"]
                )
                for item in hidden["initial_targets"]
            )
        elif group == "query":
            output.extend(
                canonical_to_position[int(item["register_index"])]
                for item in hidden["query_targets"]
            )
        else:
            raise RuntimeError("OQB1 target group differs")
    return output


def _execution_score(
    packets: Sequence[EpisodeLawPacket | None],
    public: Sequence[Mapping[str, Any]],
    assessor: Sequence[Mapping[str, Any]],
    program_predictions: Sequence[Sequence[int]],
    initial_states: Sequence[Sequence[int] | None],
    query_predictions: Sequence[int],
    *,
    state_table_key: str,
    canonical_key: str,
    reverse_state_table: bool,
) -> dict[str, Any]:
    program_cursor = query_cursor = 0
    state_exact = answer_exact = programs = queries = abstained = 0
    by_depth: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for packet, visible, hidden in zip(packets, public, assessor, strict=True):
        table = _table(visible[state_table_key], reverse_state_table)
        canonical = tuple(str(value) for value in hidden[canonical_key])
        position_to_canonical = tuple(canonical.index(value) for value in table)
        terminal_by_id = {
            str(item["program_id"]): tuple(
                int(value) for value in item["terminal_state"]
            )
            for item in hidden["transfer"]
        }
        predicted_by_id: dict[str, tuple[int, int] | None] = {}
        for transfer, hidden_transfer in zip(
            visible["transfer"], hidden["transfer"], strict=True
        ):
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
            if program_id != str(hidden_transfer["program_id"]):
                raise RuntimeError("OQB1 transfer identity differs")
            predicted_by_id[program_id] = prediction
            canonical_prediction = None
            if prediction is not None:
                values = [0, 0]
                for position, value in enumerate(prediction):
                    values[position_to_canonical[position]] = value
                canonical_prediction = tuple(values)
            gold = terminal_by_id[program_id]
            exact = canonical_prediction == gold
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
                raise RuntimeError("OQB1 query identity differs")
            prediction = predicted_by_id[program_id]
            predicted_position = int(query_predictions[query_cursor])
            gold_register = int(hidden_query["register_index"])
            gold_answer = terminal_by_id[program_id][gold_register]
            answer_exact += (
                prediction is not None
                and predicted_position in (0, 1)
                and prediction[predicted_position] == gold_answer
            )
            queries += 1
            query_cursor += 1
    if program_cursor != len(program_predictions) or query_cursor != len(
        query_predictions
    ):
        raise RuntimeError("OQB1 execution cursor differs")
    if tuple(sorted(by_depth)) != tuple(sorted(TRANSFER_DEPTHS)):
        raise RuntimeError("OQB1 transfer depth board differs")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("eal2", "ncp1"):
        parser.add_argument(f"--{name}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{name}-checkpoint-sha256", required=True)
        parser.add_argument(f"--{name}-development-report", type=Path, required=True)
        parser.add_argument(f"--{name}-development-report-sha256", required=True)
    for name in ("treatment", "control"):
        parser.add_argument(f"--{name}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{name}-checkpoint-sha256", required=True)
        parser.add_argument(f"--{name}-report", type=Path, required=True)
        parser.add_argument(f"--{name}-report-sha256", required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--public-data-sha256", required=True)
    parser.add_argument("--assessor-data", type=Path, required=True)
    parser.add_argument("--assessor-data-sha256", required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--data-report-sha256", required=True)
    parser.add_argument("--board-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing OQB1 evaluation")

    eal2_parent = _load_report(
        args.eal2_development_report, args.eal2_development_report_sha256
    )
    ncp1_parent = _load_report(
        args.ncp1_development_report, args.ncp1_development_report_sha256
    )
    treatment_report = _load_report(args.treatment_report, args.treatment_report_sha256)
    control_report = _load_report(args.control_report, args.control_report_sha256)
    data_report = _load_report(args.data_report, args.data_report_sha256)
    data_entry = data_report.get("files", {}).get(args.board_label, {})
    if (
        eal2_parent.get("status") != "pass"
        or eal2_parent.get("checkpoint_sha256") != args.eal2_checkpoint_sha256
        or ncp1_parent.get("status") != "pass"
        or ncp1_parent["parent_eal2"]["checkpoint_sha256"]
        != args.eal2_checkpoint_sha256
        or ncp1_parent["parent_eal2"]["development_report_sha256"]
        != args.eal2_development_report_sha256
        or ncp1_parent["training"]["treatment_checkpoint_sha256"]
        != args.ncp1_checkpoint_sha256
        or treatment_report.get("status") != "complete"
        or treatment_report.get("arm") != "treatment"
        or treatment_report.get("checkpoint_sha256") != args.treatment_checkpoint_sha256
        or control_report.get("status") != "complete"
        or control_report.get("arm") != "broken_quotient"
        or control_report.get("checkpoint_sha256") != args.control_checkpoint_sha256
        or data_report.get("schema") != DATA_REPORT_SCHEMA
        or not data_report.get("zero_source_name_and_identity_overlap")
        or Path(str(data_entry.get("public", {}).get("path", ""))).name
        != args.public_data.name
        or data_entry.get("public", {}).get("sha256") != args.public_data_sha256
        or Path(str(data_entry.get("assessor", {}).get("path", ""))).name
        != args.assessor_data.name
        or data_entry.get("assessor", {}).get("sha256") != args.assessor_data_sha256
    ):
        raise SystemExit("OQB1 parent/training custody differs")
    public = _load_jsonl(args.public_data, args.public_data_sha256)
    assessor = _load_jsonl(args.assessor_data, args.assessor_data_sha256)
    if len(public) != DEVELOPMENT_EPISODES or len(assessor) != DEVELOPMENT_EPISODES:
        raise SystemExit("OQB1 evaluation board geometry differs")
    for visible, hidden in zip(public, assessor, strict=True):
        validate_evaluation_episode(visible, hidden)

    board_seeds = {int(episode["seed"]) for episode in public}
    board_serials = {int(episode["serial"]) for episode in public}
    if len(board_seeds) != 1 or board_serials != set(range(DEVELOPMENT_EPISODES)):
        raise SystemExit("OQB1 board seed/serial geometry differs")
    board_seed = next(iter(board_seeds))
    if int(data_report["split_reports"][args.board_label]["seed"]) != board_seed:
        raise SystemExit("OQB1 board seed receipt differs")
    if not torch.cuda.is_available():
        raise SystemExit("OQB1 evaluation requires CUDA")
    device = torch.device("cuda")
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
    if (
        treatment_checkpoint.get("source_commit") != args.source_commit
        or control_checkpoint.get("source_commit") != args.source_commit
        or treatment_report.get("source_commit") != args.source_commit
        or control_report.get("source_commit") != args.source_commit
        or treatment_checkpoint.get("data_sha256")
        != treatment_report.get("data_sha256")
        or control_checkpoint.get("data_sha256") != control_report.get("data_sha256")
        or treatment_checkpoint.get("model_state_sha256") != treatment_hash
        or control_checkpoint.get("model_state_sha256") != control_hash
        or treatment_report.get("data_report_sha256") != args.data_report_sha256
        or control_report.get("data_report_sha256") != args.data_report_sha256
    ):
        raise SystemExit("OQB1 checkpoint/report binding differs")
    programs = _predict_programs(
        pointer, public, device=device, batch_size=args.batch_size
    )
    gold_programs = [
        tuple(int(value) for value in item["targets"])
        for episode in assessor
        for item in episode["command_targets"]
    ]
    program_report = _program_score(programs, gold_programs)

    arm_specs = {
        "treatment": {
            "model": treatment,
            "evidence_text_key": "source_text",
            "initial_text_key": "initial_text",
            "query_text_key": "query_text",
            "table_key": "register_table",
            "canonical_key": "canonical_registers",
            "hash_key": "source_sha256",
        },
        "renamed": {
            "model": treatment,
            "evidence_text_key": "renamed_source_text",
            "initial_text_key": "renamed_initial_text",
            "query_text_key": "renamed_query_text",
            "table_key": "renamed_register_table",
            "canonical_key": "canonical_renamed_registers",
            "hash_key": "renamed_source_sha256",
        },
        "table_permutation": {
            "model": treatment,
            "evidence_text_key": "source_text",
            "initial_text_key": "initial_text",
            "query_text_key": "query_text",
            "table_key": "register_table",
            "canonical_key": "canonical_registers",
            "hash_key": "source_sha256",
            "evidence_reverse": True,
            "initial_reverse": True,
            "query_reverse": True,
        },
        "cross_owner_permutation": {
            "model": treatment,
            "evidence_text_key": "source_text",
            "initial_text_key": "initial_text",
            "query_text_key": "query_text",
            "table_key": "register_table",
            "canonical_key": "canonical_registers",
            "hash_key": "source_sha256",
            "evidence_reverse": True,
        },
        "source_scrub": {
            "model": treatment,
            "evidence_text_key": "register_scrubbed_text",
            "initial_text_key": "register_scrubbed_initial_text",
            "query_text_key": "register_scrubbed_query_text",
            "table_key": "register_table",
            "canonical_key": "canonical_registers",
            "hash_key": "register_scrubbed_sha256",
        },
        "occurrence_break": {
            "model": treatment,
            "evidence_text_key": "source_text",
            "initial_text_key": "initial_text",
            "query_text_key": "query_text",
            "table_key": "register_table",
            "canonical_key": "canonical_registers",
            "hash_key": "source_sha256",
            "mode": "broken",
        },
        "broken_quotient_model": {
            "model": control,
            "evidence_text_key": "source_text",
            "initial_text_key": "initial_text",
            "query_text_key": "query_text",
            "table_key": "register_table",
            "canonical_key": "canonical_registers",
            "hash_key": "source_sha256",
        },
    }
    arms = {}
    for name, spec in arm_specs.items():
        evidence_reverse = bool(spec.get("evidence_reverse", False))
        initial_reverse = bool(spec.get("initial_reverse", False))
        query_reverse = bool(spec.get("query_reverse", False))
        predictions = _predict_arm(
            spec["model"],
            public,
            device=device,
            batch_size=args.batch_size,
            evidence_text_key=spec["evidence_text_key"],
            initial_text_key=spec["initial_text_key"],
            query_text_key=spec["query_text_key"],
            table_key=spec["table_key"],
            evidence_reverse=evidence_reverse,
            initial_reverse=initial_reverse,
            query_reverse=query_reverse,
            mode=spec.get("mode", "coherent"),
        )
        temporal = _predict_temporal(
            reader,
            predictions["evidence_records"],
            predictions["evidence"],
            device=device,
            batch_size=args.batch_size,
        )
        roles = _compose_roles(temporal, predictions["evidence"])
        owner_hash = hashlib.sha256(
            (
                reader_hash
                + (control_hash if name == "broken_quotient_model" else treatment_hash)
            ).encode()
        ).hexdigest()
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
        evidence_gold = _position_targets(
            public,
            assessor,
            group="evidence",
            table_key=spec["table_key"],
            canonical_key=spec["canonical_key"],
            reverse_table=evidence_reverse,
        )
        initial_gold = _position_targets(
            public,
            assessor,
            group="initial",
            table_key=spec["table_key"],
            canonical_key=spec["canonical_key"],
            reverse_table=initial_reverse,
        )
        query_gold = _position_targets(
            public,
            assessor,
            group="query",
            table_key=spec["table_key"],
            canonical_key=spec["canonical_key"],
            reverse_table=query_reverse,
        )
        temporal_gold = [
            tuple(int(value) // 2 for value in item["numeric_role_ids"])
            for episode in assessor
            for item in episode["evidence"]
        ]
        complete_gold = [
            tuple(
                int(time_value) * 2 + int(position_value)
                for time_value, position_value in zip(
                    temporal_values, position_values, strict=True
                )
            )
            for temporal_values, position_values in zip(
                temporal_gold, evidence_gold, strict=True
            )
        ]
        arms[name] = {
            "evidence_register": _sequence_score(
                predictions["evidence"], evidence_gold
            ),
            "temporal": _sequence_score(
                [() if value is None else value for value in temporal], temporal_gold
            ),
            "complete_roles": _sequence_score(
                [() if value is None else value for value in roles], complete_gold
            ),
            "initial_register": _sequence_score(predictions["initial"], initial_gold),
            "query_register": _scalar_score(predictions["query"], query_gold),
            "law_commits": sum(packet is not None for packet in packets),
            "law_episodes": len(packets),
            "execution": _execution_score(
                packets,
                public,
                assessor,
                programs,
                initial_states,
                predictions["query"],
                state_table_key=spec["table_key"],
                canonical_key=spec["canonical_key"],
                reverse_state_table=initial_reverse,
            ),
        }

    runtime_path = Path(__file__).with_name("diverge_oqb1_runtime.py")
    runtime_source = runtime_path.read_text()
    source_audit = {
        "runtime": str(runtime_path),
        "runtime_sha256": hashlib.sha256(runtime_source.encode()).hexdigest(),
        "contains_local_role_scanner": any(
            value in runtime_source
            for value in ("scan_register_ids", "previous_end", "nearest_register")
        ),
        "uses_declared_occurrence_quotient": all(
            value in runtime_source
            for value in (
                "exact_occurrence_quotient",
                "register_table",
                "SLOT_MARKERS",
                "re.finditer",
            )
        ),
        "contains_learned_raw_name_similarity": any(
            value in runtime_source
            for value in (
                "register_encoder",
                "register_projection",
                "source_to_register",
            )
        ),
        "uses_token_level_query_evidence": "torch.logsumexp" in runtime_source,
        "uses_exact_numeric_span_carrier": "scan_integer_spans" in runtime_source,
        "typed_carriers_absent": all(
            "initial_state" not in item and "symbols" not in item
            for episode in public
            for item in episode["transfer"]
        )
        and all(
            "register_index" not in item
            for episode in public
            for item in episode["queries"]
        ),
    }
    positive = ("treatment", "renamed", "table_permutation")
    conditions = {
        "qualified_parents_pass": True,
        "program_at_least_99_percent": program_report["exact_rate"] >= 0.99,
        "program_depth_floor_at_least_95_percent": program_report["minimum_depth_rate"]
        >= 0.95,
        "all_positive_components_at_least_99_percent": all(
            min(
                arms[name]["evidence_register"]["exact_rate"],
                arms[name]["temporal"]["exact_rate"],
                arms[name]["complete_roles"]["exact_rate"],
                arms[name]["initial_register"]["exact_rate"],
                arms[name]["query_register"]["rate"],
                arms[name]["law_commits"] / arms[name]["law_episodes"],
            )
            >= 0.99
            for name in positive
        ),
        "all_positive_execution_at_least_99_percent": all(
            min(
                arms[name]["execution"]["state_exact_rate"],
                arms[name]["execution"]["answer_exact_rate"],
            )
            >= 0.99
            for name in positive
        ),
        "all_positive_depth_floors_at_least_95_percent": all(
            arms[name]["execution"]["minimum_depth_rate"] >= 0.95 for name in positive
        ),
        "cross_owner_permutation_collapses": arms["cross_owner_permutation"][
            "execution"
        ]["state_exact_rate"]
        <= 0.05
        and arms["cross_owner_permutation"]["execution"]["answer_exact_rate"] <= 0.55,
        "source_scrub_fails_closed": arms["source_scrub"]["evidence_register"][
            "exact_rate"
        ]
        <= 0.01
        and arms["source_scrub"]["initial_register"]["exact_rate"] <= 0.01
        and arms["source_scrub"]["query_register"]["rate"] <= 0.01
        and arms["source_scrub"]["execution"]["state_exact_rate"] <= 0.01
        and arms["source_scrub"]["execution"]["answer_exact_rate"] <= 0.01,
        "occurrence_break_within_calibrated_chance": arms["occurrence_break"][
            "evidence_register"
        ]["exact_rate"]
        <= 0.30
        and arms["occurrence_break"]["initial_register"]["exact_rate"] <= 0.35
        and arms["occurrence_break"]["query_register"]["rate"] <= 0.55
        and arms["occurrence_break"]["execution"]["state_exact_rate"] <= 0.20
        and arms["occurrence_break"]["execution"]["answer_exact_rate"] <= 0.35,
        "broken_model_within_calibrated_chance": arms["broken_quotient_model"][
            "evidence_register"
        ]["exact_rate"]
        <= 0.30
        and arms["broken_quotient_model"]["initial_register"]["exact_rate"] <= 0.35
        and arms["broken_quotient_model"]["query_register"]["rate"] <= 0.55
        and arms["broken_quotient_model"]["execution"]["state_exact_rate"] <= 0.20
        and arms["broken_quotient_model"]["execution"]["answer_exact_rate"] <= 0.35,
        "matched_training": treatment_report["initial_state_sha256"]
        == control_report["initial_state_sha256"]
        and treatment_report["data_sha256"] == control_report["data_sha256"]
        and treatment_report["updates"] == control_report["updates"]
        and treatment_report["batch_size"] == control_report["batch_size"]
        and treatment_report["learning_rate"] == control_report["learning_rate"]
        and treatment_report["charged_examples"] == control_report["charged_examples"]
        and treatment_checkpoint["config"] == control_checkpoint["config"],
        "checkpoint_report_custody": treatment_checkpoint["model_state_sha256"]
        == treatment_report["final_state_sha256"]
        and control_checkpoint["model_state_sha256"]
        == control_report["final_state_sha256"]
        and treatment_report["checkpoint_sha256"] == args.treatment_checkpoint_sha256
        and control_report["checkpoint_sha256"] == args.control_checkpoint_sha256,
        "parent_weights_bit_identical": reader_hash
        == eal2_parent["reader_state_sha256"]
        and args.ncp1_checkpoint_sha256
        == ncp1_parent["training"]["treatment_checkpoint_sha256"],
        "runtime_uses_only_declared_occurrence_identity": source_audit[
            "uses_declared_occurrence_quotient"
        ]
        and not source_audit["contains_local_role_scanner"]
        and not source_audit["contains_learned_raw_name_similarity"],
        "token_query_and_typed_deletion": source_audit[
            "uses_token_level_query_evidence"
        ]
        and source_audit["typed_carriers_absent"],
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
            "report": str(args.data_report),
            "report_sha256": args.data_report_sha256,
            "board_label": args.board_label,
            "board_seed": board_seed,
            "board_identity_sha256": canonical_sha256(
                [episode["identity_sha256"] for episode in public]
            ),
            "public": str(args.public_data),
            "public_sha256": args.public_data_sha256,
            "assessor": str(args.assessor_data),
            "assessor_sha256": args.assessor_data_sha256,
        },
        "program": program_report,
        "arms": arms,
        "source_audit": source_audit,
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "status": report["status"],
                "state": arms["treatment"]["execution"]["state_exact_rate"],
                "answer": arms["treatment"]["execution"]["answer_exact_rate"],
                "renamed": arms["renamed"]["execution"]["answer_exact_rate"],
                "permuted": arms["table_permutation"]["execution"]["answer_exact_rate"],
                "cross_owner": arms["cross_owner_permutation"]["execution"][
                    "answer_exact_rate"
                ],
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
