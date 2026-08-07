#!/usr/bin/env python3
"""Evaluate spanless value events composed with frozen neural law synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from build_diverge_snl1_data import REPORT_SCHEMA as DATA_REPORT_SCHEMA
from diverge_eal1_data import canonical_sha256
from diverge_eal1_runtime import EpisodeLawPacket, module_state_sha256, sha256_path
from diverge_ncp1_runtime import load_pointer
from diverge_nls1_runtime import load_synthesizer
from diverge_oqb1_runtime import load_binder
from diverge_snl1_runtime import CompileControl, compile_neural_event_laws
from diverge_sve1_data import DEVELOPMENT_EPISODES, validate_evaluation_episode
from diverge_sve1_runtime import digit_scrub, load_transducer
from eval_diverge_eal1 import _load_jsonl
from eval_diverge_jrb1 import _predict_programs, _program_score
from eval_diverge_oqb1 import _execution_score
from eval_diverge_sve1 import (
    _decode_initial_states,
    _gold_evidence_events,
    _gold_initial_events,
    _gold_queries,
    _predict_arm,
    _scalar_score,
    _sequence_score,
)


SCHEMA = "shohin-diverge-snl1-evaluation-v1"


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
        raise RuntimeError(f"SNL1 report hash differs: {path}")
    return json.loads(path.read_text())


def _compile_packets(
    public: Sequence[Mapping[str, Any]],
    events: Sequence[Sequence[int]],
    model: torch.nn.Module,
    *,
    device: torch.device,
    text_key: str,
    hash_key: str,
    event_owner_sha256: str,
    law_owner_sha256: str,
    scrub_values: bool,
    law_control: CompileControl,
) -> list[EpisodeLawPacket | None]:
    stride = len(public[0]["evidence"])
    if len(events) != len(public) * stride:
        raise RuntimeError("SNL1 complete-event count differs")
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
        compilation = compile_neural_event_laws(
            {"aliases": episode["aliases"], "evidence": visible_evidence},
            events[episode_index * stride : (episode_index + 1) * stride],
            model,  # type: ignore[arg-type]
            device=device,
            event_owner_sha256=event_owner_sha256,
            model_owner_sha256=law_owner_sha256,
            text_key="source_text",
            hash_key="source_sha256",
            control=law_control,
        )
        output.append(compilation.packet)
    return output


def _law_score(
    packets: Sequence[EpisodeLawPacket | None],
    assessor: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exact = 0
    rows = 0
    row_exact = 0
    for packet, hidden in zip(packets, assessor, strict=True):
        gold = tuple(
            tuple(tuple(int(value) for value in row) for row in matrix)
            for matrix in hidden["matrices"]
        )
        if packet is not None:
            exact += int(packet.rows == gold)
            for predicted_matrix, gold_matrix in zip(packet.rows, gold, strict=True):
                for predicted, target in zip(
                    predicted_matrix, gold_matrix, strict=True
                ):
                    row_exact += int(predicted == target)
        rows += len(gold) * 2
    return {
        "exact": exact,
        "total": len(assessor),
        "exact_rate": exact / len(assessor),
        "row_exact": row_exact,
        "rows": rows,
        "row_rate": row_exact / rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("oqb1", "ncp1"):
        parser.add_argument(f"--{name}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{name}-checkpoint-sha256", required=True)
        parser.add_argument(f"--{name}-development-report", type=Path, required=True)
        parser.add_argument(f"--{name}-development-report-sha256", required=True)
    for name in ("sve1-treatment", "sve1-control", "nls1-treatment", "nls1-control"):
        parser.add_argument(f"--{name}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{name}-checkpoint-sha256", required=True)
        parser.add_argument(f"--{name}-report", type=Path, required=True)
        parser.add_argument(f"--{name}-report-sha256", required=True)
    parser.add_argument("--sve1-development-report", type=Path, required=True)
    parser.add_argument("--sve1-development-report-sha256", required=True)
    parser.add_argument("--nls1-development-report", type=Path, required=True)
    parser.add_argument("--nls1-development-report-sha256", required=True)
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
        raise SystemExit("refusing existing SNL1 evaluation")

    oqb1_parent = _load_report(
        args.oqb1_development_report, args.oqb1_development_report_sha256
    )
    ncp1_parent = _load_report(
        args.ncp1_development_report, args.ncp1_development_report_sha256
    )
    sve1_parent = _load_report(
        args.sve1_development_report, args.sve1_development_report_sha256
    )
    nls1_parent = _load_report(
        args.nls1_development_report, args.nls1_development_report_sha256
    )
    reports = {
        name: _load_report(
            getattr(args, f"{name.replace('-', '_')}_report"),
            getattr(args, f"{name.replace('-', '_')}_report_sha256"),
        )
        for name in (
            "sve1-treatment",
            "sve1-control",
            "nls1-treatment",
            "nls1-control",
        )
    }
    data_report = _load_report(args.data_report, args.data_report_sha256)
    data_entry = data_report.get("files", {}).get(args.board_label, {})
    if (
        oqb1_parent.get("status") != "pass"
        or ncp1_parent.get("status") != "pass"
        or sve1_parent.get("status") != "pass"
        or nls1_parent.get("status") != "fail"
        or nls1_parent.get("arms", {})
        .get("treatment", {})
        .get("law", {})
        .get("exact_rate")
        != 1.0
        or data_report.get("schema") != DATA_REPORT_SCHEMA
        or not data_report.get("zero_source_name_and_identity_overlap")
        or data_entry.get("public", {}).get("sha256") != args.public_data_sha256
        or data_entry.get("assessor", {}).get("sha256") != args.assessor_data_sha256
    ):
        raise SystemExit("SNL1 parent/data custody differs")
    public = _load_jsonl(args.public_data, args.public_data_sha256)
    assessor = _load_jsonl(args.assessor_data, args.assessor_data_sha256)
    if len(public) != DEVELOPMENT_EPISODES or len(assessor) != DEVELOPMENT_EPISODES:
        raise SystemExit("SNL1 board geometry differs")
    for visible, hidden in zip(public, assessor, strict=True):
        validate_evaluation_episode(visible, hidden)
    board_seed = int(public[0]["seed"])
    if (
        any(int(item["seed"]) != board_seed for item in public)
        or int(data_report["split_reports"][args.board_label]["seed"]) != board_seed
    ):
        raise SystemExit("SNL1 board seed receipt differs")
    if not torch.cuda.is_available():
        raise SystemExit("SNL1 evaluation requires CUDA")
    device = torch.device("cuda")

    query_model, _ = load_binder(args.oqb1_checkpoint, args.oqb1_checkpoint_sha256)
    pointer, _ = load_pointer(args.ncp1_checkpoint, args.ncp1_checkpoint_sha256)
    sve1_treatment, _ = load_transducer(
        args.sve1_treatment_checkpoint, args.sve1_treatment_checkpoint_sha256
    )
    sve1_control, _ = load_transducer(
        args.sve1_control_checkpoint, args.sve1_control_checkpoint_sha256
    )
    nls1_treatment, _ = load_synthesizer(
        args.nls1_treatment_checkpoint, args.nls1_treatment_checkpoint_sha256
    )
    nls1_control, _ = load_synthesizer(
        args.nls1_control_checkpoint, args.nls1_control_checkpoint_sha256
    )
    models = (
        query_model,
        pointer,
        sve1_treatment,
        sve1_control,
        nls1_treatment,
        nls1_control,
    )
    for model in models:
        model.to(device).eval()
    hashes = {
        "query": module_state_sha256(query_model),
        "pointer": module_state_sha256(pointer),
        "sve1_treatment": module_state_sha256(sve1_treatment),
        "sve1_control": module_state_sha256(sve1_control),
        "nls1_treatment": module_state_sha256(nls1_treatment),
        "nls1_control": module_state_sha256(nls1_control),
    }

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
        "shuffled_event_model": {"event_model": sve1_control},
        "shuffled_law_model": {"law_model": nls1_control},
        "outcome_scrub": {"law_control": "scrub_outcomes"},
        "one_example": {"law_control": "one_example"},
        "law_reset": {"reset_laws": True},
    }
    arms = {}
    for name, overrides in arm_specs.items():
        spec = {
            "event_model": sve1_treatment,
            "law_model": nls1_treatment,
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
            "law_control": "normal",
            "reset_laws": False,
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
        event_hash = (
            hashes["sve1_control"]
            if name == "shuffled_event_model"
            else hashes["sve1_treatment"]
        )
        law_hash = (
            hashes["nls1_control"]
            if name == "shuffled_law_model"
            else hashes["nls1_treatment"]
        )
        packets = _compile_packets(
            public,
            predictions["evidence"],
            spec["law_model"],
            device=device,
            text_key=spec["evidence_text_key"],
            hash_key=spec["hash_key"],
            event_owner_sha256=event_hash,
            law_owner_sha256=law_hash,
            scrub_values=spec["scrub_values"],
            law_control=spec["law_control"],
        )
        if spec["reset_laws"]:
            packets = [None] * len(packets)
        initial_states = _decode_initial_states(predictions["initial"])
        arms[name] = {
            "evidence_events": _sequence_score(
                predictions["evidence"],
                _gold_evidence_events(
                    public,
                    assessor,
                    table_key=spec["table_key"],
                    canonical_key=spec["canonical_key"],
                    reverse_table=spec["evidence_reverse"],
                ),
            ),
            "initial_events": _sequence_score(
                predictions["initial"],
                _gold_initial_events(
                    public,
                    assessor,
                    table_key=spec["table_key"],
                    canonical_key=spec["canonical_key"],
                    reverse_table=spec["initial_reverse"],
                ),
            ),
            "query_register": _scalar_score(
                predictions["query"],
                _gold_queries(
                    public,
                    assessor,
                    table_key=spec["table_key"],
                    canonical_key=spec["canonical_key"],
                    reverse_table=spec["query_reverse"],
                ),
            ),
            "law": _law_score(packets, assessor),
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

    runtime_path = Path(__file__).with_name("diverge_snl1_runtime.py")
    runtime_source = runtime_path.read_text()
    source_audit = {
        "runtime": str(runtime_path),
        "runtime_sha256": hashlib.sha256(runtime_source.encode()).hexdigest(),
        "uses_neural_event_laws": "compile_neural_event_laws" in runtime_source,
        "uses_exact_support_solver": any(
            token in runtime_source
            for token in ("compile_event_laws", "empty_support", "set(range")
        ),
        "uses_numeric_span_scanner": "scan_integer_spans" in runtime_source,
        "uses_raw_integer_parser": any(
            token in runtime_source for token in ("int(text[", "_INTEGER")
        ),
    }
    positive = ("treatment", "renamed", "table_reindex")
    conditions = {
        "qualified_semantic_parents_pass": True,
        "nls1_prior_failure_preserved": nls1_parent.get("status") == "fail",
        "program_at_least_99_percent": program_report["exact_rate"] >= 0.99,
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
        "cross_owner_reindex_collapses": arms["cross_owner_reindex"]["execution"][
            "state_exact_rate"
        ]
        <= 0.05
        and arms["cross_owner_reindex"]["execution"]["answer_exact_rate"] <= 0.10,
        "value_scrub_fails_closed": arms["value_scrub"]["execution"]["state_exact_rate"]
        <= 0.01
        and arms["value_scrub"]["execution"]["answer_exact_rate"] <= 0.01,
        "occurrence_break_fails_closed": arms["occurrence_break"]["execution"][
            "state_exact_rate"
        ]
        <= 0.20
        and arms["occurrence_break"]["execution"]["answer_exact_rate"] <= 0.35,
        "shuffled_event_model_fails_closed": arms["shuffled_event_model"]["execution"][
            "state_exact_rate"
        ]
        <= 0.05,
        "shuffled_law_model_fails_closed": arms["shuffled_law_model"]["execution"][
            "state_exact_rate"
        ]
        <= 0.05,
        "outcome_scrub_fails_closed": arms["outcome_scrub"]["execution"][
            "state_exact_rate"
        ]
        <= 0.05,
        "one_example_is_underdetermined": arms["one_example"]["execution"][
            "state_exact_rate"
        ]
        <= 0.20,
        "law_reset_fails_closed": arms["law_reset"]["execution"]["state_exact_rate"]
        == 0.0,
        "parent_weights_bit_identical": hashes["query"]
        == oqb1_parent["training"]["treatment_state_sha256"]
        and args.ncp1_checkpoint_sha256
        == ncp1_parent["training"]["treatment_checkpoint_sha256"]
        and args.sve1_treatment_checkpoint_sha256
        == sve1_parent["training"]["treatment_checkpoint_sha256"]
        and reports["nls1-treatment"]["final_state_sha256"] == hashes["nls1_treatment"]
        and reports["nls1-control"]["final_state_sha256"] == hashes["nls1_control"],
        "matched_nls1_training": reports["nls1-treatment"]["initial_state_sha256"]
        == reports["nls1-control"]["initial_state_sha256"]
        and reports["nls1-treatment"]["data_sha256"]
        == reports["nls1-control"]["data_sha256"]
        and reports["nls1-treatment"]["updates"] == reports["nls1-control"]["updates"]
        and reports["nls1-treatment"]["charged_examples"]
        == reports["nls1-control"]["charged_examples"],
        "spanless_neural_runtime": source_audit["uses_neural_event_laws"]
        and not source_audit["uses_exact_support_solver"]
        and not source_audit["uses_numeric_span_scanner"]
        and not source_audit["uses_raw_integer_parser"],
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
            "sve1_development_report_sha256": args.sve1_development_report_sha256,
            "nls1_development_report_sha256": args.nls1_development_report_sha256,
            "state_sha256": hashes,
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
