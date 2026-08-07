#!/usr/bin/env python3
"""Evaluate one frozen DIVERGE-SVE1 spanless value-event gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_eal1_data import canonical_sha256
from diverge_eal1_runtime import EpisodeLawPacket, module_state_sha256, sha256_path
from diverge_ncp1_runtime import load_pointer
from diverge_oqb1_runtime import (
    QuotientMode,
    load_binder,
    tensorize_quotient_sources,
)
from diverge_sve1_data import (
    DEVELOPMENT_EPISODES,
    REPORT_SCHEMA as DATA_REPORT_SCHEMA,
    VALUES,
    validate_evaluation_episode,
)
from diverge_sve1_runtime import (
    EVIDENCE_BLANK_ID,
    INITIAL_BLANK_ID,
    compile_event_laws,
    decode_initial_events,
    digit_scrub,
    greedy_ctc_decode,
    load_transducer,
    tensorize_event_sources,
)
from eval_diverge_eal1 import _load_jsonl
from eval_diverge_jrb1 import (
    _predict_programs,
    _program_score,
    _scalar_score,
    _sequence_score,
)
from eval_diverge_oqb1 import _execution_score


SCHEMA = "shohin-diverge-sve1-evaluation-v1"


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
        raise RuntimeError(f"SVE1 report hash differs: {path}")
    return json.loads(path.read_text())


def _table(values: Sequence[str], reverse: bool) -> tuple[str, str]:
    table = tuple(str(value) for value in values)
    if len(table) != 2 or len(set(table)) != 2:
        raise RuntimeError("SVE1 evaluator table differs")
    return (table[1], table[0]) if reverse else (table[0], table[1])


def _records(
    public: Sequence[Mapping[str, Any]],
    *,
    group: str,
    text_key: str,
    table_key: str,
) -> list[dict[str, Any]]:
    output = []
    for episode in public:
        for index, item in enumerate(episode[group]):
            output.append(
                {
                    "source_text": item[text_key],
                    "register_table": list(episode[table_key]),
                    "serial": f"{episode['episode_id']}|{group}|{index}",
                }
            )
    return output


@torch.no_grad()
def _predict_events(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    kind: str,
    reverse_table: bool,
    quotient_mode: QuotientMode,
    scrub_values: bool,
) -> list[tuple[int, ...]]:
    expected = (2, 2) if kind == "evidence" else (1, 1)
    blank = EVIDENCE_BLANK_ID if kind == "evidence" else INITIAL_BLANK_ID
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        tensors = tensorize_event_sources(
            batch,
            device,
            text_key="source_text",
            expected_occurrences=expected,
            quotient_mode=quotient_mode,
            reverse_table=reverse_table,
            scrub_values=scrub_values,
        )
        decoded = greedy_ctc_decode(
            model(tensors[0], tensors[1], kind=kind),  # type: ignore[attr-defined]
            tensors[2],
            blank_id=blank,
        )
        output.extend(
            value if bool(valid) else ()
            for value, valid in zip(decoded, tensors[3].cpu(), strict=True)
        )
    return output


@torch.no_grad()
def _predict_queries(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    reverse_table: bool,
    mode: QuotientMode,
) -> list[int]:
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        tensors = tensorize_quotient_sources(
            batch,
            device,
            text_key="source_text",
            table_key="register_table",
            mention_count=None,
            mode=mode,
            reverse_table=reverse_table,
        )
        predictions = model.forward_query(tensors[0], tensors[1]).argmax(dim=-1)  # type: ignore[attr-defined]
        output.extend(
            int(value) if bool(valid) else -1
            for value, valid in zip(
                predictions.detach().cpu(), tensors[4].cpu(), strict=True
            )
        )
    return output


def _predict_arm(
    event_model: torch.nn.Module,
    query_model: torch.nn.Module,
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
    quotient_mode: QuotientMode = "coherent",
    scrub_values: bool = False,
) -> dict[str, Any]:
    evidence_records = _records(
        public,
        group="evidence",
        text_key=evidence_text_key,
        table_key=table_key,
    )
    initial_records = _records(
        public,
        group="transfer",
        text_key=initial_text_key,
        table_key=table_key,
    )
    query_records = _records(
        public,
        group="queries",
        text_key=query_text_key,
        table_key=table_key,
    )
    return {
        "evidence": _predict_events(
            event_model,
            evidence_records,
            device=device,
            batch_size=batch_size,
            kind="evidence",
            reverse_table=evidence_reverse,
            quotient_mode=quotient_mode,
            scrub_values=scrub_values,
        ),
        "initial": _predict_events(
            event_model,
            initial_records,
            device=device,
            batch_size=batch_size,
            kind="initial",
            reverse_table=initial_reverse,
            quotient_mode=quotient_mode,
            scrub_values=scrub_values,
        ),
        "query": _predict_queries(
            query_model,
            query_records,
            device=device,
            batch_size=batch_size,
            reverse_table=query_reverse,
            mode=quotient_mode,
        ),
    }


def _canonical_to_position(
    public: Mapping[str, Any],
    assessor: Mapping[str, Any],
    *,
    table_key: str,
    canonical_key: str,
    reverse_table: bool,
) -> tuple[int, int]:
    table = _table(public[table_key], reverse_table)
    canonical = tuple(str(value) for value in assessor[canonical_key])
    return tuple(table.index(value) for value in canonical)  # type: ignore[return-value]


def _gold_evidence_events(
    public: Sequence[Mapping[str, Any]],
    assessor: Sequence[Mapping[str, Any]],
    *,
    table_key: str,
    canonical_key: str,
    reverse_table: bool,
) -> list[tuple[int, ...]]:
    output = []
    for visible, hidden in zip(public, assessor, strict=True):
        positions = _canonical_to_position(
            visible,
            hidden,
            table_key=table_key,
            canonical_key=canonical_key,
            reverse_table=reverse_table,
        )
        for item in hidden["evidence"]:
            values = (*item["before"], *item["after"])
            sequence = []
            for role in item["numeric_role_ids"]:
                role = int(role)
                complete = (role // 2) * 2 + positions[role % 2]
                sequence.append(complete * VALUES + int(values[role]))
            output.append(tuple(sequence))
    return output


def _gold_initial_events(
    public: Sequence[Mapping[str, Any]],
    assessor: Sequence[Mapping[str, Any]],
    *,
    table_key: str,
    canonical_key: str,
    reverse_table: bool,
) -> list[tuple[int, ...]]:
    output = []
    for visible, hidden in zip(public, assessor, strict=True):
        positions = _canonical_to_position(
            visible,
            hidden,
            table_key=table_key,
            canonical_key=canonical_key,
            reverse_table=reverse_table,
        )
        for item in hidden["initial_targets"]:
            state = tuple(int(value) for value in item["state"])
            output.append(
                tuple(
                    positions[int(register)] * VALUES + state[int(register)]
                    for register in item["mention_register_targets"]
                )
            )
    return output


def _gold_queries(
    public: Sequence[Mapping[str, Any]],
    assessor: Sequence[Mapping[str, Any]],
    *,
    table_key: str,
    canonical_key: str,
    reverse_table: bool,
) -> list[int]:
    output = []
    for visible, hidden in zip(public, assessor, strict=True):
        positions = _canonical_to_position(
            visible,
            hidden,
            table_key=table_key,
            canonical_key=canonical_key,
            reverse_table=reverse_table,
        )
        output.extend(
            positions[int(item["register_index"])] for item in hidden["query_targets"]
        )
    return output


def _compile_packets(
    public: Sequence[Mapping[str, Any]],
    events: Sequence[Sequence[int]],
    *,
    text_key: str,
    hash_key: str,
    owner_state_sha256: str,
    scrub_values: bool,
) -> list[EpisodeLawPacket | None]:
    stride = len(public[0]["evidence"])
    if len(events) != len(public) * stride:
        raise RuntimeError("SVE1 complete-event count differs")
    output = []
    for episode_index, episode in enumerate(public):
        visible_evidence = []
        for item in episode["evidence"]:
            text = str(item[text_key])
            if scrub_values:
                text = digit_scrub(text)
            visible_evidence.append(
                {
                    "source_text": text,
                    "source_sha256": hashlib.sha256(text.encode()).hexdigest()
                    if scrub_values
                    else str(item[hash_key]),
                }
            )
        visible = {"aliases": episode["aliases"], "evidence": visible_evidence}
        compilation = compile_event_laws(
            visible,
            events[episode_index * stride : (episode_index + 1) * stride],
            owner_state_sha256=owner_state_sha256,
            text_key="source_text",
            hash_key="source_sha256",
        )
        output.append(compilation.packet)
    return output


def _decode_initial_states(
    sequences: Sequence[Sequence[int]],
) -> list[tuple[int, int] | None]:
    output = []
    for sequence in sequences:
        try:
            output.append(decode_initial_events(sequence))
        except RuntimeError:
            output.append(None)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("oqb1", "ncp1"):
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
        raise SystemExit("refusing existing SVE1 evaluation")

    oqb1_parent = _load_report(
        args.oqb1_development_report, args.oqb1_development_report_sha256
    )
    ncp1_parent = _load_report(
        args.ncp1_development_report, args.ncp1_development_report_sha256
    )
    treatment_report = _load_report(args.treatment_report, args.treatment_report_sha256)
    control_report = _load_report(args.control_report, args.control_report_sha256)
    data_report = _load_report(args.data_report, args.data_report_sha256)
    data_entry = data_report.get("files", {}).get(args.board_label, {})
    if (
        oqb1_parent.get("status") != "pass"
        or oqb1_parent["training"]["treatment_checkpoint_sha256"]
        != args.oqb1_checkpoint_sha256
        or ncp1_parent.get("status") != "pass"
        or ncp1_parent["training"]["treatment_checkpoint_sha256"]
        != args.ncp1_checkpoint_sha256
        or treatment_report.get("status") != "complete"
        or treatment_report.get("arm") != "treatment"
        or treatment_report.get("checkpoint_sha256") != args.treatment_checkpoint_sha256
        or control_report.get("status") != "complete"
        or control_report.get("arm") != "shuffled_targets"
        or control_report.get("checkpoint_sha256") != args.control_checkpoint_sha256
        or data_report.get("schema") != DATA_REPORT_SCHEMA
        or not data_report.get("zero_source_name_and_identity_overlap")
        or data_entry.get("public", {}).get("sha256") != args.public_data_sha256
        or data_entry.get("assessor", {}).get("sha256") != args.assessor_data_sha256
    ):
        raise SystemExit("SVE1 parent/training custody differs")
    public = _load_jsonl(args.public_data, args.public_data_sha256)
    assessor = _load_jsonl(args.assessor_data, args.assessor_data_sha256)
    if len(public) != DEVELOPMENT_EPISODES or len(assessor) != DEVELOPMENT_EPISODES:
        raise SystemExit("SVE1 evaluation board geometry differs")
    for visible, hidden in zip(public, assessor, strict=True):
        validate_evaluation_episode(visible, hidden)
    board_seeds = {int(episode["seed"]) for episode in public}
    if len(board_seeds) != 1:
        raise SystemExit("SVE1 board seed geometry differs")
    board_seed = next(iter(board_seeds))
    if int(data_report["split_reports"][args.board_label]["seed"]) != board_seed:
        raise SystemExit("SVE1 board seed receipt differs")
    if not torch.cuda.is_available():
        raise SystemExit("SVE1 evaluation requires CUDA")
    device = torch.device("cuda")
    query_model, oqb1_checkpoint = load_binder(
        args.oqb1_checkpoint, args.oqb1_checkpoint_sha256
    )
    pointer, _ = load_pointer(args.ncp1_checkpoint, args.ncp1_checkpoint_sha256)
    treatment, treatment_checkpoint = load_transducer(
        args.treatment_checkpoint, args.treatment_checkpoint_sha256
    )
    control, control_checkpoint = load_transducer(
        args.control_checkpoint, args.control_checkpoint_sha256
    )
    query_model = query_model.to(device).eval()
    pointer = pointer.to(device).eval()
    treatment = treatment.to(device).eval()
    control = control.to(device).eval()
    query_hash = module_state_sha256(query_model)
    pointer_hash = module_state_sha256(pointer)
    treatment_hash = module_state_sha256(treatment)
    control_hash = module_state_sha256(control)
    if (
        treatment_checkpoint.get("source_commit") != args.source_commit
        or control_checkpoint.get("source_commit") != args.source_commit
        or treatment_report.get("source_commit") != args.source_commit
        or control_report.get("source_commit") != args.source_commit
        or treatment_checkpoint.get("model_state_sha256") != treatment_hash
        or control_checkpoint.get("model_state_sha256") != control_hash
        or treatment_report.get("data_report_sha256") != args.data_report_sha256
        or control_report.get("data_report_sha256") != args.data_report_sha256
    ):
        raise SystemExit("SVE1 checkpoint/report binding differs")

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
        "treatment": {},
        "renamed": {
            "evidence_text_key": "renamed_source_text",
            "initial_text_key": "renamed_initial_text",
            "query_text_key": "renamed_query_text",
            "table_key": "renamed_register_table",
            "canonical_key": "canonical_renamed_registers",
            "hash_key": "renamed_source_sha256",
        },
        "table_reindex": {
            "evidence_reverse": True,
            "initial_reverse": True,
            "query_reverse": True,
        },
        "cross_owner_reindex": {"evidence_reverse": True},
        "value_scrub": {"scrub_values": True},
        "occurrence_break": {"quotient_mode": "broken"},
        "shuffled_target_model": {"event_model": control},
    }
    arms = {}
    for name, overrides in arm_specs.items():
        spec = {
            "event_model": treatment,
            "evidence_text_key": "source_text",
            "initial_text_key": "initial_text",
            "query_text_key": "query_text",
            "table_key": "register_table",
            "canonical_key": "canonical_registers",
            "hash_key": "source_sha256",
            "evidence_reverse": False,
            "initial_reverse": False,
            "query_reverse": False,
            "quotient_mode": "coherent",
            "scrub_values": False,
            **overrides,
        }
        predictions = _predict_arm(
            spec["event_model"],
            query_model,
            public,
            device=device,
            batch_size=args.batch_size,
            evidence_text_key=spec["evidence_text_key"],
            initial_text_key=spec["initial_text_key"],
            query_text_key=spec["query_text_key"],
            table_key=spec["table_key"],
            evidence_reverse=spec["evidence_reverse"],
            initial_reverse=spec["initial_reverse"],
            query_reverse=spec["query_reverse"],
            quotient_mode=spec["quotient_mode"],
            scrub_values=spec["scrub_values"],
        )
        owner_hash = hashlib.sha256(
            (
                query_hash
                + (control_hash if name == "shuffled_target_model" else treatment_hash)
            ).encode()
        ).hexdigest()
        packets = _compile_packets(
            public,
            predictions["evidence"],
            text_key=spec["evidence_text_key"],
            hash_key=spec["hash_key"],
            owner_state_sha256=owner_hash,
            scrub_values=spec["scrub_values"],
        )
        initial_states = _decode_initial_states(predictions["initial"])
        evidence_gold = _gold_evidence_events(
            public,
            assessor,
            table_key=spec["table_key"],
            canonical_key=spec["canonical_key"],
            reverse_table=spec["evidence_reverse"],
        )
        initial_gold = _gold_initial_events(
            public,
            assessor,
            table_key=spec["table_key"],
            canonical_key=spec["canonical_key"],
            reverse_table=spec["initial_reverse"],
        )
        query_gold = _gold_queries(
            public,
            assessor,
            table_key=spec["table_key"],
            canonical_key=spec["canonical_key"],
            reverse_table=spec["query_reverse"],
        )
        arms[name] = {
            "evidence_events": _sequence_score(predictions["evidence"], evidence_gold),
            "initial_events": _sequence_score(predictions["initial"], initial_gold),
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
                reverse_state_table=spec["initial_reverse"],
            ),
        }

    runtime_path = Path(__file__).with_name("diverge_sve1_runtime.py")
    runtime_source = runtime_path.read_text()
    source_audit = {
        "runtime": str(runtime_path),
        "runtime_sha256": hashlib.sha256(runtime_source.encode()).hexdigest(),
        "uses_numeric_span_scanner": "scan_integer_spans" in runtime_source,
        "uses_raw_integer_parser": any(
            value in runtime_source
            for value in ("int(text[", "_INTEGER", 'finditer(r"\\d')
        ),
        "uses_complete_value_events": all(
            value in runtime_source
            for value in (
                "SpanlessValueEventTransducer",
                "compile_event_laws",
                "decode_initial_events",
                "greedy_ctc_decode",
            )
        ),
        "uses_occurrence_quotient": "exact_occurrence_quotient" in runtime_source,
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
    positive = ("treatment", "renamed", "table_reindex")
    conditions = {
        "qualified_parents_pass": True,
        "program_at_least_99_percent": program_report["exact_rate"] >= 0.99,
        "all_positive_components_at_least_99_percent": all(
            min(
                arms[name]["evidence_events"]["exact_rate"],
                arms[name]["initial_events"]["exact_rate"],
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
        "cross_owner_reindex_collapses": arms["cross_owner_reindex"]["execution"][
            "state_exact_rate"
        ]
        <= 0.05
        and arms["cross_owner_reindex"]["execution"]["answer_exact_rate"] <= 0.10,
        "value_scrub_fails_closed": arms["value_scrub"]["evidence_events"]["exact_rate"]
        <= 0.01
        and arms["value_scrub"]["initial_events"]["exact_rate"] <= 0.01
        and arms["value_scrub"]["execution"]["state_exact_rate"] <= 0.01
        and arms["value_scrub"]["execution"]["answer_exact_rate"] <= 0.01,
        "occurrence_break_within_chance": arms["occurrence_break"]["evidence_events"][
            "exact_rate"
        ]
        <= 0.30
        and arms["occurrence_break"]["initial_events"]["exact_rate"] <= 0.35
        and arms["occurrence_break"]["execution"]["state_exact_rate"] <= 0.20
        and arms["occurrence_break"]["execution"]["answer_exact_rate"] <= 0.35,
        "shuffled_model_fails_closed": arms["shuffled_target_model"]["evidence_events"][
            "exact_rate"
        ]
        <= 0.01
        and arms["shuffled_target_model"]["initial_events"]["exact_rate"] <= 0.01
        and arms["shuffled_target_model"]["execution"]["state_exact_rate"] <= 0.05
        and arms["shuffled_target_model"]["execution"]["answer_exact_rate"] <= 0.10,
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
        == control_report["final_state_sha256"],
        "parent_weights_bit_identical": query_hash
        == oqb1_parent["training"]["treatment_state_sha256"]
        and args.ncp1_checkpoint_sha256
        == ncp1_parent["training"]["treatment_checkpoint_sha256"],
        "spanless_runtime_and_typed_deletion": not source_audit[
            "uses_numeric_span_scanner"
        ]
        and not source_audit["uses_raw_integer_parser"]
        and source_audit["uses_complete_value_events"]
        and source_audit["uses_occurrence_quotient"]
        and source_audit["typed_carriers_absent"],
    }
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "source_commit": args.source_commit,
        "parents": {
            "oqb1_checkpoint_sha256": args.oqb1_checkpoint_sha256,
            "oqb1_development_report_sha256": args.oqb1_development_report_sha256,
            "ncp1_checkpoint_sha256": args.ncp1_checkpoint_sha256,
            "ncp1_development_report_sha256": args.ncp1_development_report_sha256,
            "query_state_sha256": query_hash,
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
                "evidence": arms["treatment"]["evidence_events"]["exact_rate"],
                "initial": arms["treatment"]["initial_events"]["exact_rate"],
                "state": arms["treatment"]["execution"]["state_exact_rate"],
                "answer": arms["treatment"]["execution"]["answer_exact_rate"],
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
