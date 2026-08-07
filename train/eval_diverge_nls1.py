#!/usr/bin/env python3
"""Evaluate one frozen DIVERGE-NLS1 neural law-synthesis gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_eal1_runtime import EpisodeLawPacket, module_state_sha256, sha256_path
from diverge_eal2_data import DEVELOPMENT_EPISODES, validate_episode
from diverge_eal2_runtime import load_reader
from diverge_nls1_runtime import compile_neural_laws, load_synthesizer
from eval_diverge_eal1 import _load_jsonl, _score_packets
from eval_diverge_eal2 import _predict, _reader_score


SCHEMA = "shohin-diverge-nls1-evaluation-v1"


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_training_report(path: Path, expected_sha256: str, arm: str) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("NLS1 training report hash differs")
    report = json.loads(path.read_text())
    if report.get("status") != "complete" or report.get("arm") != arm:
        raise RuntimeError("NLS1 training report custody differs")
    return report


def _row_score(
    packets: Sequence[EpisodeLawPacket | None], assessor: Sequence[Mapping[str, Any]]
) -> dict[str, float | int]:
    exact = 0
    matrices = 0
    rows = 0
    matrix_exact = 0
    for packet, hidden in zip(packets, assessor, strict=True):
        gold = tuple(
            tuple(tuple(int(value) for value in row) for row in matrix)
            for matrix in hidden["matrices"]
        )
        predicted = None if packet is None else packet.rows
        if predicted is not None:
            exact += sum(
                predicted_row == gold_row
                for predicted_matrix, gold_matrix in zip(predicted, gold, strict=True)
                for predicted_row, gold_row in zip(
                    predicted_matrix, gold_matrix, strict=True
                )
            )
            matrix_exact += sum(
                predicted_matrix == gold_matrix
                for predicted_matrix, gold_matrix in zip(predicted, gold, strict=True)
            )
        matrices += len(gold)
        rows += len(gold) * 2
    return {
        "row_exact": exact,
        "rows": rows,
        "row_rate": exact / rows,
        "matrix_exact": matrix_exact,
        "matrices": matrices,
        "matrix_rate": matrix_exact / matrices,
    }


def _compile_all(
    public: Sequence[Mapping[str, Any]],
    assignments: Sequence[Sequence[int]],
    model: torch.nn.Module,
    *,
    device: torch.device,
    reader_state_sha256: str,
    text_key: str,
    control: str,
) -> list[EpisodeLawPacket | None]:
    stride = len(public[0]["evidence"])
    packets = []
    for index, visible in enumerate(public):
        compilation = compile_neural_laws(
            visible,
            assignments[index * stride : (index + 1) * stride],
            model,  # type: ignore[arg-type]
            device=device,
            reader_state_sha256=reader_state_sha256,
            text_key=text_key,
            control=control,
        )
        packets.append(compilation.packet)
    return packets


def _arm_report(
    packets: Sequence[EpisodeLawPacket | None],
    public: Sequence[Mapping[str, Any]],
    assessor: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "law": _row_score(packets, assessor),
        "execution": _score_packets(packets, public, assessor),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eal2-checkpoint", type=Path, required=True)
    parser.add_argument("--eal2-checkpoint-sha256", required=True)
    parser.add_argument("--eal2-development-report", type=Path, required=True)
    parser.add_argument("--eal2-development-report-sha256", required=True)
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
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing NLS1 evaluation")
    if sha256_path(args.eal2_development_report) != args.eal2_development_report_sha256:
        raise SystemExit("NLS1 parent EAL2 report hash differs")
    parent = json.loads(args.eal2_development_report.read_text())
    if (
        parent.get("status") != "pass"
        or parent.get("checkpoint_sha256") != args.eal2_checkpoint_sha256
    ):
        raise SystemExit("NLS1 parent EAL2 gate differs")
    treatment_report = _load_training_report(
        args.treatment_report, args.treatment_report_sha256, "treatment"
    )
    control_report = _load_training_report(
        args.control_report, args.control_report_sha256, "shuffled_outcomes"
    )
    public = _load_jsonl(args.public_data, args.public_data_sha256)
    assessor = _load_jsonl(args.assessor_data, args.assessor_data_sha256)
    if len(public) != DEVELOPMENT_EPISODES or len(assessor) != DEVELOPMENT_EPISODES:
        raise SystemExit("NLS1 evaluation board geometry differs")
    for visible, hidden in zip(public, assessor, strict=True):
        validate_episode(visible, hidden)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader, _ = load_reader(args.eal2_checkpoint, args.eal2_checkpoint_sha256)
    treatment, treatment_checkpoint = load_synthesizer(
        args.treatment_checkpoint, args.treatment_checkpoint_sha256
    )
    control, control_checkpoint = load_synthesizer(
        args.control_checkpoint, args.control_checkpoint_sha256
    )
    reader = reader.to(device).eval()
    treatment = treatment.to(device).eval()
    control = control.to(device).eval()
    reader_hash = module_state_sha256(reader)
    records = [item for episode in public for item in episode["evidence"]]
    hidden_evidence = [item for episode in assessor for item in episode["evidence"]]
    normal_gold = [
        tuple(int(value) // 2 for value in item["numeric_role_ids"])
        for item in hidden_evidence
    ]
    counterfactual_gold = [
        tuple(int(value) // 2 for value in item["counterfactual_role_ids"])
        for item in hidden_evidence
    ]
    normal = _predict(
        reader,
        records,
        text_key="source_text",
        device=device,
        batch_size=args.batch_size,
    )
    counterfactual = _predict(
        reader,
        records,
        text_key="counterfactual_text",
        device=device,
        batch_size=args.batch_size,
    )
    scrubbed = _predict(
        reader,
        records,
        text_key="scrubbed_text",
        device=device,
        batch_size=args.batch_size,
    )
    reader_report = {
        "normal": _reader_score(records, normal, normal_gold),
        "counterfactual": _reader_score(records, counterfactual, counterfactual_gold),
        "temporal_scrub": _reader_score(records, scrubbed, normal_gold),
    }

    packets = {
        "treatment": _compile_all(
            public,
            normal,
            treatment,
            device=device,
            reader_state_sha256=reader_hash,
            text_key="source_text",
            control="normal",
        ),
        "counterfactual": _compile_all(
            public,
            counterfactual,
            treatment,
            device=device,
            reader_state_sha256=reader_hash,
            text_key="counterfactual_text",
            control="normal",
        ),
        "temporal_scrub": _compile_all(
            public,
            scrubbed,
            treatment,
            device=device,
            reader_state_sha256=reader_hash,
            text_key="scrubbed_text",
            control="normal",
        ),
        "one_example": _compile_all(
            public,
            normal,
            treatment,
            device=device,
            reader_state_sha256=reader_hash,
            text_key="source_text",
            control="one_example",
        ),
        "scrub_outcomes": _compile_all(
            public,
            normal,
            treatment,
            device=device,
            reader_state_sha256=reader_hash,
            text_key="source_text",
            control="scrub_outcomes",
        ),
        "shuffled_outcome_model": _compile_all(
            public,
            normal,
            control,
            device=device,
            reader_state_sha256=reader_hash,
            text_key="source_text",
            control="normal",
        ),
    }
    arms = {
        name: _arm_report(value, public, assessor) for name, value in packets.items()
    }
    runtime_path = Path(__file__).with_name("diverge_nls1_runtime.py")
    runtime_source = runtime_path.read_text()
    source_audit = {
        "runtime": str(runtime_path),
        "runtime_sha256": hashlib.sha256(runtime_source.encode()).hexdigest(),
        "contains_exact_support_intersection": "supports[" in runtime_source
        or "compatible_rows" in runtime_source,
        "imports_exact_apply_matrix": "apply_matrix" in runtime_source,
        "bounded_row_vocabulary_declared": "ROW_CANDIDATES" in runtime_source,
    }
    conditions = {
        "normal_reader_at_least_99_percent": reader_report["normal"][
            "complete_exact_rate"
        ]
        >= 0.99,
        "counterfactual_reader_at_least_99_percent": reader_report["counterfactual"][
            "complete_exact_rate"
        ]
        >= 0.99,
        "temporal_scrub_at_most_30_percent": reader_report["temporal_scrub"][
            "complete_exact_rate"
        ]
        <= 0.30,
        "treatment_rows_at_least_99_percent": arms["treatment"]["law"]["row_rate"]
        >= 0.99,
        "treatment_state_at_least_99_percent": arms["treatment"]["execution"][
            "state_exact_rate"
        ]
        >= 0.99,
        "treatment_query_at_least_99_percent": arms["treatment"]["execution"][
            "query_exact_rate"
        ]
        >= 0.99,
        "treatment_depth_floor_at_least_95_percent": arms["treatment"]["execution"][
            "minimum_depth_rate"
        ]
        >= 0.95,
        "counterfactual_state_at_least_99_percent": arms["counterfactual"]["execution"][
            "state_exact_rate"
        ]
        >= 0.99,
        "shuffled_model_state_at_most_5_percent": arms["shuffled_outcome_model"][
            "execution"
        ]["state_exact_rate"]
        <= 0.05,
        "scrubbed_outcomes_state_at_most_5_percent": arms["scrub_outcomes"][
            "execution"
        ]["state_exact_rate"]
        <= 0.05,
        "one_example_state_at_most_20_percent": arms["one_example"]["execution"][
            "state_exact_rate"
        ]
        <= 0.20,
        "temporal_scrub_state_at_most_10_percent": arms["temporal_scrub"]["execution"][
            "state_exact_rate"
        ]
        <= 0.10,
        "matched_initialization_data_and_schedule": treatment_report[
            "initial_state_sha256"
        ]
        == control_report["initial_state_sha256"]
        and treatment_report["data_sha256"] == control_report["data_sha256"]
        and treatment_report["updates"] == control_report["updates"]
        and treatment_report["batch_size"] == control_report["batch_size"],
        "checkpoint_report_custody_matches": treatment_checkpoint["model_state_sha256"]
        == treatment_report["final_state_sha256"]
        and control_checkpoint["model_state_sha256"]
        == control_report["final_state_sha256"],
        "runtime_has_no_exact_support_solver": not source_audit[
            "contains_exact_support_intersection"
        ]
        and not source_audit["imports_exact_apply_matrix"],
        "sources_deleted_before_execution": all(
            packet is not None and not hasattr(packet, "source_text")
            for packet in packets["treatment"]
        ),
    }
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "source_commit": args.source_commit,
        "parent_eal2": {
            "checkpoint_sha256": args.eal2_checkpoint_sha256,
            "development_report_sha256": args.eal2_development_report_sha256,
            "reader_state_sha256": reader_hash,
        },
        "training": {
            "treatment_checkpoint_sha256": args.treatment_checkpoint_sha256,
            "treatment_report_sha256": args.treatment_report_sha256,
            "control_checkpoint_sha256": args.control_checkpoint_sha256,
            "control_report_sha256": args.control_report_sha256,
        },
        "data": {
            "public": str(args.public_data),
            "public_sha256": args.public_data_sha256,
            "assessor": str(args.assessor_data),
            "assessor_sha256": args.assessor_data_sha256,
        },
        "reader": reader_report,
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
                "row_rate": arms["treatment"]["law"]["row_rate"],
                "state_rate": arms["treatment"]["execution"]["state_exact_rate"],
                "query_rate": arms["treatment"]["execution"]["query_exact_rate"],
                "shuffled_state": arms["shuffled_outcome_model"]["execution"][
                    "state_exact_rate"
                ],
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
