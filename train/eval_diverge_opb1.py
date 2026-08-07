#!/usr/bin/env python3
"""Evaluate learned evidence-operation binding through the frozen SNL1 stack."""

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
from diverge_ncp1_runtime import (
    greedy_ctc_decode as greedy_program_decode,
    load_pointer,
    tensorize_commands,
)
from diverge_nls1_runtime import load_synthesizer
from diverge_opb1_data import (
    DEVELOPMENT_EPISODES,
    REPORT_SCHEMA as DATA_REPORT_SCHEMA,
    rotate_aliases,
    validate_evaluation_episode,
)
from diverge_opb1_runtime import (
    compile_pointer_event_laws,
    load_operation_pointer,
    tensorize_operation_sources,
)
from diverge_oqb1_runtime import load_binder
from diverge_sve1_runtime import load_transducer
from eval_diverge_eal1 import _load_jsonl
from eval_diverge_ncp1 import _program_score
from eval_diverge_oqb1 import _execution_score
from eval_diverge_snl1 import _law_score
from eval_diverge_sve1 import (
    _decode_initial_states,
    _gold_evidence_events,
    _gold_initial_events,
    _gold_queries,
    _predict_arm,
    _scalar_score,
    _sequence_score,
)


SCHEMA = "shohin-diverge-opb1-evaluation-v1"


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
        raise RuntimeError(f"OPB1 report hash differs: {path}")
    return json.loads(path.read_text())


def _alias_tables(
    public: Sequence[Mapping[str, Any]], kind: str
) -> list[tuple[str, ...]]:
    if kind == "original":
        return [tuple(str(value) for value in episode["aliases"]) for episode in public]
    if kind == "renamed":
        return [
            tuple(str(value) for value in episode["renamed_aliases"])
            for episode in public
        ]
    if kind == "rotated":
        return [rotate_aliases(episode["aliases"]) for episode in public]
    raise RuntimeError("OPB1 alias-table kind differs")


def _evidence_records(
    public: Sequence[Mapping[str, Any]], *, text_key: str, alias_kind: str
) -> list[dict[str, Any]]:
    tables = _alias_tables(public, alias_kind)
    return [
        {"source_text": item[text_key], "aliases": list(table)}
        for episode, table in zip(public, tables, strict=True)
        for item in episode["evidence"]
    ]


@torch.no_grad()
def _predict_operations(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> list[int]:
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        logits = model(*tensorize_operation_sources(batch, device))
        output.extend(int(value) for value in logits.argmax(dim=-1).detach().cpu())
    return output


def _command_records(
    public: Sequence[Mapping[str, Any]], *, text_key: str, alias_kind: str
) -> list[dict[str, Any]]:
    tables = _alias_tables(public, alias_kind)
    return [
        {"source_text": transfer[text_key], "aliases": list(table)}
        for episode, table in zip(public, tables, strict=True)
        for transfer in episode["transfer"]
    ]


@torch.no_grad()
def _predict_programs(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> list[tuple[int, ...]]:
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        command_ids, command_mask, alias_ids, alias_mask, lengths = tensorize_commands(
            batch, device
        )
        output.extend(
            greedy_program_decode(
                model(command_ids, command_mask, alias_ids, alias_mask), lengths
            )
        )
    return output


def _operation_gold(
    assessor: Sequence[Mapping[str, Any]], *, rotated: bool
) -> list[int]:
    return [
        (int(target) - 1) % 8 if rotated else int(target)
        for episode in assessor
        for target in episode["operation_targets"]
    ]


def _program_gold(
    assessor: Sequence[Mapping[str, Any]], *, rotated: bool
) -> list[tuple[int, ...]]:
    return [
        tuple(
            (int(value) - 1) % 8 if rotated else int(value) for value in item["targets"]
        )
        for episode in assessor
        for item in episode["command_targets"]
    ]


def _arm_public(
    public: Sequence[Mapping[str, Any]], *, alias_kind: str
) -> list[dict[str, Any]]:
    tables = _alias_tables(public, alias_kind)
    return [
        {**episode, "aliases": list(table)}
        for episode, table in zip(public, tables, strict=True)
    ]


def _compile_packets(
    public: Sequence[Mapping[str, Any]],
    events: Sequence[Sequence[int]],
    operations: Sequence[int],
    law_model: torch.nn.Module,
    *,
    alias_kind: str,
    hash_key: str,
    device: torch.device,
    event_owner_sha256: str,
    pointer_owner_sha256: str,
    law_owner_sha256: str,
) -> list[EpisodeLawPacket | None]:
    stride = len(public[0]["evidence"])
    if len(events) != len(public) * stride or len(operations) != len(events):
        raise RuntimeError("OPB1 event/operation count differs")
    tables = _alias_tables(public, alias_kind)
    output = []
    for episode_index, (episode, aliases) in enumerate(
        zip(public, tables, strict=True)
    ):
        start = episode_index * stride
        end = start + stride
        compilation = compile_pointer_event_laws(
            aliases,
            [str(item[hash_key]) for item in episode["evidence"]],
            events[start:end],
            operations[start:end],
            law_model,  # type: ignore[arg-type]
            device=device,
            event_owner_sha256=event_owner_sha256,
            pointer_owner_sha256=pointer_owner_sha256,
            law_owner_sha256=law_owner_sha256,
        )
        output.append(compilation.packet)
    return output


def _law_score_for_aliases(
    packets: Sequence[EpisodeLawPacket | None],
    public: Sequence[Mapping[str, Any]],
    assessor: Sequence[Mapping[str, Any]],
    *,
    alias_kind: str,
    table_key: str,
    canonical_key: str,
) -> dict[str, Any]:
    if alias_kind in ("original", "renamed"):
        return _law_score(
            packets,
            public,
            assessor,
            table_key=table_key,
            canonical_key=canonical_key,
            reverse_table=False,
        )
    reordered_assessor = []
    for hidden in assessor:
        item = dict(hidden)
        item["matrices"] = list(hidden["matrices"])[1:] + [hidden["matrices"][0]]
        reordered_assessor.append(item)
    return _law_score(
        packets,
        public,
        reordered_assessor,
        table_key=table_key,
        canonical_key=canonical_key,
        reverse_table=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snl1-development-report", type=Path, required=True)
    parser.add_argument("--snl1-development-report-sha256", required=True)
    parser.add_argument("--snl1-data-report-sha256", required=True)
    for name in ("oqb1", "ncp1", "sve1", "nls1"):
        parser.add_argument(f"--{name}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{name}-checkpoint-sha256", required=True)
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
        raise SystemExit("refusing existing OPB1 evaluation")

    snl1 = _load_report(
        args.snl1_development_report, args.snl1_development_report_sha256
    )
    treatment_report = _load_report(args.treatment_report, args.treatment_report_sha256)
    control_report = _load_report(args.control_report, args.control_report_sha256)
    data_report = _load_report(args.data_report, args.data_report_sha256)
    data_entry = data_report.get("files", {}).get(args.board_label, {})
    if (
        snl1.get("status") != "pass"
        or snl1.get("parents", {}).get("oqb1_checkpoint_sha256")
        != args.oqb1_checkpoint_sha256
        or snl1.get("parents", {}).get("ncp1_checkpoint_sha256")
        != args.ncp1_checkpoint_sha256
        or data_report.get("schema") != DATA_REPORT_SCHEMA
        or not data_report.get("zero_source_name_and_identity_overlap")
        or data_report.get("parent_snl1", {}).get("report_sha256")
        != args.snl1_data_report_sha256
        or data_entry.get("public", {}).get("sha256") != args.public_data_sha256
        or data_entry.get("assessor", {}).get("sha256") != args.assessor_data_sha256
        or treatment_report.get("status") != "complete"
        or treatment_report.get("arm") != "treatment"
        or treatment_report.get("checkpoint_sha256") != args.treatment_checkpoint_sha256
        or control_report.get("status") != "complete"
        or control_report.get("arm") != "decoy_table"
        or control_report.get("checkpoint_sha256") != args.control_checkpoint_sha256
    ):
        raise SystemExit("OPB1 parent/data/training custody differs")
    public = _load_jsonl(args.public_data, args.public_data_sha256)
    assessor = _load_jsonl(args.assessor_data, args.assessor_data_sha256)
    if len(public) != DEVELOPMENT_EPISODES or len(assessor) != DEVELOPMENT_EPISODES:
        raise SystemExit("OPB1 board geometry differs")
    for visible, hidden in zip(public, assessor, strict=True):
        validate_evaluation_episode(visible, hidden)
    board_seed = int(public[0]["seed"])
    if (
        any(int(episode["seed"]) != board_seed for episode in public)
        or int(data_report["split_reports"][args.board_label]["seed"]) != board_seed
    ):
        raise SystemExit("OPB1 board seed receipt differs")
    if not torch.cuda.is_available():
        raise SystemExit("OPB1 evaluation requires CUDA")
    device = torch.device("cuda")

    query_model, _ = load_binder(args.oqb1_checkpoint, args.oqb1_checkpoint_sha256)
    command_model, _ = load_pointer(args.ncp1_checkpoint, args.ncp1_checkpoint_sha256)
    event_model, _ = load_transducer(args.sve1_checkpoint, args.sve1_checkpoint_sha256)
    law_model, _ = load_synthesizer(args.nls1_checkpoint, args.nls1_checkpoint_sha256)
    treatment, treatment_checkpoint = load_operation_pointer(
        args.treatment_checkpoint, args.treatment_checkpoint_sha256
    )
    control, control_checkpoint = load_operation_pointer(
        args.control_checkpoint, args.control_checkpoint_sha256
    )
    models = (query_model, command_model, event_model, law_model, treatment, control)
    for model in models:
        model.to(device).eval()
    hashes = {
        "query": module_state_sha256(query_model),
        "command": module_state_sha256(command_model),
        "event": module_state_sha256(event_model),
        "law": module_state_sha256(law_model),
        "treatment": module_state_sha256(treatment),
        "control": module_state_sha256(control),
    }

    operation_predictions = {
        "treatment": _predict_operations(
            treatment,
            _evidence_records(public, text_key="source_text", alias_kind="original"),
            device=device,
            batch_size=args.batch_size,
        ),
        "renamed": _predict_operations(
            treatment,
            _evidence_records(
                public,
                text_key="fully_renamed_source_text",
                alias_kind="renamed",
            ),
            device=device,
            batch_size=args.batch_size,
        ),
        "rotated": _predict_operations(
            treatment,
            _evidence_records(public, text_key="source_text", alias_kind="rotated"),
            device=device,
            batch_size=args.batch_size,
        ),
        "operation_scrub": _predict_operations(
            treatment,
            _evidence_records(
                public, text_key="operation_scrubbed_text", alias_kind="original"
            ),
            device=device,
            batch_size=args.batch_size,
        ),
        "decoy_model": _predict_operations(
            control,
            _evidence_records(public, text_key="source_text", alias_kind="original"),
            device=device,
            batch_size=args.batch_size,
        ),
    }
    program_predictions = {
        "original": _predict_programs(
            command_model,
            _command_records(public, text_key="command_text", alias_kind="original"),
            device=device,
            batch_size=args.batch_size,
        ),
        "renamed": _predict_programs(
            command_model,
            _command_records(
                public, text_key="renamed_command_text", alias_kind="renamed"
            ),
            device=device,
            batch_size=args.batch_size,
        ),
        "rotated": _predict_programs(
            command_model,
            _command_records(public, text_key="command_text", alias_kind="rotated"),
            device=device,
            batch_size=args.batch_size,
        ),
    }
    program_reports = {
        "original": _program_score(
            program_predictions["original"], _program_gold(assessor, rotated=False)
        ),
        "renamed": _program_score(
            program_predictions["renamed"], _program_gold(assessor, rotated=False)
        ),
        "rotated": _program_score(
            program_predictions["rotated"], _program_gold(assessor, rotated=True)
        ),
    }

    positive_original = _predict_arm(
        event_model,
        query_model,
        public,
        device=device,
        batch_size=args.batch_size,
        evidence_text_key="source_text",
        initial_text_key="initial_text",
        query_text_key="query_text",
        table_key="register_table",
    )
    positive_renamed = _predict_arm(
        event_model,
        query_model,
        public,
        device=device,
        batch_size=args.batch_size,
        evidence_text_key="fully_renamed_source_text",
        initial_text_key="renamed_initial_text",
        query_text_key="renamed_query_text",
        table_key="renamed_register_table",
    )
    scrubbed = _predict_arm(
        event_model,
        query_model,
        public,
        device=device,
        batch_size=args.batch_size,
        evidence_text_key="operation_scrubbed_text",
        initial_text_key="initial_text",
        query_text_key="query_text",
        table_key="register_table",
    )

    arm_specs = {
        "treatment": {
            "operations": operation_predictions["treatment"],
            "operation_rotated": False,
            "alias_kind": "original",
            "programs": program_predictions["original"],
            "predictions": positive_original,
            "hash_key": "source_sha256",
            "table_key": "register_table",
            "canonical_key": "canonical_registers",
            "pointer_hash": hashes["treatment"],
        },
        "renamed": {
            "operations": operation_predictions["renamed"],
            "operation_rotated": False,
            "alias_kind": "renamed",
            "programs": program_predictions["renamed"],
            "predictions": positive_renamed,
            "hash_key": "fully_renamed_source_sha256",
            "table_key": "renamed_register_table",
            "canonical_key": "canonical_renamed_registers",
            "pointer_hash": hashes["treatment"],
        },
        "alias_reindex": {
            "operations": operation_predictions["rotated"],
            "operation_rotated": True,
            "alias_kind": "rotated",
            "programs": program_predictions["rotated"],
            "predictions": positive_original,
            "hash_key": "source_sha256",
            "table_key": "register_table",
            "canonical_key": "canonical_registers",
            "pointer_hash": hashes["treatment"],
        },
        "cross_owner_reindex": {
            "operations": operation_predictions["rotated"],
            "operation_rotated": True,
            "alias_kind": "rotated",
            "programs": program_predictions["original"],
            "predictions": positive_original,
            "hash_key": "source_sha256",
            "table_key": "register_table",
            "canonical_key": "canonical_registers",
            "pointer_hash": hashes["treatment"],
        },
        "operation_scrub": {
            "operations": operation_predictions["operation_scrub"],
            "operation_rotated": False,
            "alias_kind": "original",
            "programs": program_predictions["original"],
            "predictions": scrubbed,
            "hash_key": "operation_scrubbed_sha256",
            "table_key": "register_table",
            "canonical_key": "canonical_registers",
            "pointer_hash": hashes["treatment"],
        },
        "decoy_table_model": {
            "operations": operation_predictions["decoy_model"],
            "operation_rotated": False,
            "alias_kind": "original",
            "programs": program_predictions["original"],
            "predictions": positive_original,
            "hash_key": "source_sha256",
            "table_key": "register_table",
            "canonical_key": "canonical_registers",
            "pointer_hash": hashes["control"],
        },
    }
    arms = {}
    for name, spec in arm_specs.items():
        packets = _compile_packets(
            public,
            spec["predictions"]["evidence"],
            spec["operations"],
            law_model,
            alias_kind=spec["alias_kind"],
            hash_key=spec["hash_key"],
            device=device,
            event_owner_sha256=hashes["event"],
            pointer_owner_sha256=spec["pointer_hash"],
            law_owner_sha256=hashes["law"],
        )
        visible = _arm_public(public, alias_kind=spec["alias_kind"])
        initial_states = _decode_initial_states(spec["predictions"]["initial"])
        arms[name] = {
            "operation": _scalar_score(
                spec["operations"],
                _operation_gold(assessor, rotated=spec["operation_rotated"]),
            ),
            "evidence_events": _sequence_score(
                spec["predictions"]["evidence"],
                _gold_evidence_events(
                    public,
                    assessor,
                    table_key=spec["table_key"],
                    canonical_key=spec["canonical_key"],
                    reverse_table=False,
                ),
            ),
            "initial_events": _sequence_score(
                spec["predictions"]["initial"],
                _gold_initial_events(
                    public,
                    assessor,
                    table_key=spec["table_key"],
                    canonical_key=spec["canonical_key"],
                    reverse_table=False,
                ),
            ),
            "query_register": _scalar_score(
                spec["predictions"]["query"],
                _gold_queries(
                    public,
                    assessor,
                    table_key=spec["table_key"],
                    canonical_key=spec["canonical_key"],
                    reverse_table=False,
                ),
            ),
            "law": _law_score_for_aliases(
                packets,
                public,
                assessor,
                alias_kind=spec["alias_kind"],
                table_key=spec["table_key"],
                canonical_key=spec["canonical_key"],
            ),
            "execution": _execution_score(
                packets,
                visible,
                assessor,
                spec["programs"],
                initial_states,
                spec["predictions"]["query"],
                state_table_key=spec["table_key"],
                canonical_key=spec["canonical_key"],
                reverse_state_table=False,
            ),
        }

    runtime_path = Path(__file__).with_name("diverge_opb1_runtime.py")
    runtime_source = runtime_path.read_text()
    source_audit = {
        "runtime": str(runtime_path),
        "runtime_sha256": hashlib.sha256(runtime_source.encode()).hexdigest(),
        "contains_exact_alias_search": any(
            token in runtime_source
            for token in ("re.search", "re.find", ".find(", "aliases.index")
        ),
        "uses_dynamic_alias_pointer": "torch.einsum" in runtime_source,
        "uses_pointer_selected_groups": "compile_pointer_event_laws" in runtime_source,
        "public_exposes_operation_target": any(
            "operation" in item or "operation_index" in item
            for episode in public
            for item in episode["evidence"]
        ),
    }
    positive = ("treatment", "renamed", "alias_reindex")
    conditions = {
        "snl1_parent_passes": snl1.get("status") == "pass",
        "all_positive_operation_at_least_99_percent": all(
            arms[name]["operation"]["rate"] >= 0.99 for name in positive
        ),
        "all_positive_components_at_least_99_percent": all(
            min(
                arms[name]["evidence_events"]["exact_rate"],
                arms[name]["initial_events"]["exact_rate"],
                arms[name]["query_register"]["rate"],
                arms[name]["law"]["exact_rate"],
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
        "all_program_paths_at_least_99_percent": all(
            report["exact_rate"] >= 0.99 for report in program_reports.values()
        ),
        "cross_owner_reindex_preserves_local_binding": arms["cross_owner_reindex"][
            "operation"
        ]["rate"]
        >= 0.99,
        "cross_owner_reindex_collapses_execution": arms["cross_owner_reindex"][
            "execution"
        ]["state_exact_rate"]
        <= 0.05
        and arms["cross_owner_reindex"]["execution"]["answer_exact_rate"] <= 0.10,
        "operation_scrub_collapses": arms["operation_scrub"]["operation"]["rate"]
        <= 0.20
        and arms["operation_scrub"]["execution"]["state_exact_rate"] <= 0.05,
        "decoy_table_model_collapses": arms["decoy_table_model"]["operation"]["rate"]
        <= 0.20
        and arms["decoy_table_model"]["execution"]["state_exact_rate"] <= 0.05,
        "matched_operation_training": treatment_report["initial_state_sha256"]
        == control_report["initial_state_sha256"]
        and treatment_report["data_sha256"] == control_report["data_sha256"]
        and treatment_report["updates"] == control_report["updates"]
        and treatment_report["batch_size"] == control_report["batch_size"]
        and treatment_report["charged_examples"] == control_report["charged_examples"],
        "checkpoint_report_custody_matches": treatment_checkpoint["model_state_sha256"]
        == treatment_report["final_state_sha256"]
        and control_checkpoint["model_state_sha256"]
        == control_report["final_state_sha256"],
        "snl1_parent_weights_bit_identical": hashes["query"]
        == snl1["parents"]["state_sha256"]["query"]
        and hashes["command"] == snl1["parents"]["state_sha256"]["pointer"]
        and hashes["event"] == snl1["parents"]["state_sha256"]["sve1_treatment"]
        and hashes["law"] == snl1["parents"]["state_sha256"]["nls1_treatment"],
        "learned_pointer_runtime_has_no_exact_alias_search": not source_audit[
            "contains_exact_alias_search"
        ]
        and source_audit["uses_dynamic_alias_pointer"]
        and source_audit["uses_pointer_selected_groups"],
        "operation_targets_are_assessor_only": not source_audit[
            "public_exposes_operation_target"
        ],
    }
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "source_commit": args.source_commit,
        "parents": {
            "snl1_development_report_sha256": args.snl1_development_report_sha256,
            "snl1_data_report_sha256": args.snl1_data_report_sha256,
            "state_sha256": hashes,
        },
        "training": {
            "treatment_checkpoint_sha256": args.treatment_checkpoint_sha256,
            "treatment_report_sha256": args.treatment_report_sha256,
            "control_checkpoint_sha256": args.control_checkpoint_sha256,
            "control_report_sha256": args.control_report_sha256,
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
        "program": program_reports,
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
                "operation": arms["treatment"]["operation"]["rate"],
                "law": arms["treatment"]["law"]["exact_rate"],
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
