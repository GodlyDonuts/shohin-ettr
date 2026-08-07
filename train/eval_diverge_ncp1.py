#!/usr/bin/env python3
"""Evaluate one frozen DIVERGE-NCP1 natural-command gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_eal1_runtime import EpisodeLawPacket, module_state_sha256, sha256_path
from diverge_eal2_data import DEVELOPMENT_EPISODES, TRANSFER_DEPTHS
from diverge_eal2_runtime import compile_episode_laws, load_reader
from diverge_ncp1_data import validate_evaluation_episode
from diverge_ncp1_runtime import greedy_ctc_decode, load_pointer, tensorize_commands
from eval_diverge_eal1 import _load_jsonl, _score_packets
from eval_diverge_eal2 import _predict, _reader_score


SCHEMA = "shohin-diverge-ncp1-evaluation-v1"


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
        raise RuntimeError("NCP1 training report hash differs")
    report = json.loads(path.read_text())
    if report.get("status") != "complete" or report.get("arm") != arm:
        raise RuntimeError("NCP1 training report custody differs")
    return report


def _command_records(
    public: Sequence[Mapping[str, Any]], *, text_key: str, aliases_key: str
) -> list[dict[str, Any]]:
    return [
        {"source_text": transfer[text_key], "aliases": episode[aliases_key]}
        for episode in public
        for transfer in episode["transfer"]
    ]


@torch.no_grad()
def _predict_commands(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    rotate_alias_table: bool = False,
) -> list[tuple[int, ...]]:
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        command_ids, command_mask, alias_ids, alias_mask, lengths = tensorize_commands(
            batch, device, rotate_alias_table=rotate_alias_table
        )
        output.extend(
            greedy_ctc_decode(
                model(command_ids, command_mask, alias_ids, alias_mask),  # type: ignore[operator]
                lengths,
            )
        )
    return output


def _gold_sequences(
    assessor: Sequence[Mapping[str, Any]], *, reverse: bool = False
) -> list[tuple[int, ...]]:
    key = "reverse_targets" if reverse else "targets"
    return [
        tuple(int(value) for value in target[key])
        for episode in assessor
        for target in episode["command_targets"]
    ]


def _program_score(
    predicted: Sequence[Sequence[int]],
    gold: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if len(predicted) != len(gold):
        raise RuntimeError("NCP1 prediction count differs")
    exact = 0
    token_exact = 0
    tokens = 0
    by_depth: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for guess, target in zip(predicted, gold, strict=True):
        guess_tuple = tuple(int(value) for value in guess)
        target_tuple = tuple(int(value) for value in target)
        is_exact = guess_tuple == target_tuple
        exact += is_exact
        token_exact += sum(
            left == right for left, right in zip(guess_tuple, target_tuple)
        )
        tokens += max(len(guess_tuple), len(target_tuple))
        by_depth[len(target_tuple)][0] += int(is_exact)
        by_depth[len(target_tuple)][1] += 1
    if tuple(sorted(by_depth)) != tuple(sorted(TRANSFER_DEPTHS)):
        raise RuntimeError("NCP1 transfer depth board differs")
    depth = {
        str(key): {"exact": value[0], "total": value[1], "rate": value[0] / value[1]}
        for key, value in sorted(by_depth.items())
    }
    return {
        "exact": exact,
        "total": len(gold),
        "exact_rate": exact / len(gold),
        "token_exact": token_exact,
        "tokens": tokens,
        "token_rate": token_exact / max(1, tokens),
        "by_depth": depth,
        "minimum_depth_rate": min(value["rate"] for value in depth.values()),
    }


def _candidate_public(
    public: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    output = []
    cursor = 0
    for episode in public:
        aliases = tuple(str(value) for value in episode["aliases"])
        transfers = []
        for transfer in episode["transfer"]:
            depth = int(transfer["depth"])
            values = tuple(int(value) for value in predictions[cursor])
            cursor += 1
            normalized = (values + (0,) * depth)[:depth]
            if any(value < 0 or value >= len(aliases) for value in normalized):
                raise RuntimeError("NCP1 predicted operation leaves its carrier")
            transfers.append(
                {
                    "program_id": transfer["program_id"],
                    "depth": depth,
                    "initial_state": transfer["initial_state"],
                    "symbols": [aliases[value] for value in normalized],
                }
            )
        output.append(
            {
                "aliases": list(aliases),
                "registers": episode["registers"],
                "evidence": episode["evidence"],
                "transfer": transfers,
                "queries": episode["queries"],
            }
        )
    if cursor != len(predictions):
        raise RuntimeError("NCP1 prediction cursor differs")
    return output


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
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing NCP1 evaluation")
    if sha256_path(args.eal2_development_report) != args.eal2_development_report_sha256:
        raise SystemExit("NCP1 parent EAL2 report hash differs")
    parent = json.loads(args.eal2_development_report.read_text())
    if (
        parent.get("status") != "pass"
        or parent.get("checkpoint_sha256") != args.eal2_checkpoint_sha256
    ):
        raise SystemExit("NCP1 parent EAL2 gate differs")
    treatment_report = _load_training_report(
        args.treatment_report, args.treatment_report_sha256, "treatment"
    )
    control_report = _load_training_report(
        args.control_report, args.control_report_sha256, "shuffled_table"
    )
    public = _load_jsonl(args.public_data, args.public_data_sha256)
    assessor = _load_jsonl(args.assessor_data, args.assessor_data_sha256)
    if len(public) != DEVELOPMENT_EPISODES or len(assessor) != DEVELOPMENT_EPISODES:
        raise SystemExit("NCP1 evaluation board geometry differs")
    for visible, hidden in zip(public, assessor, strict=True):
        validate_evaluation_episode(visible, hidden)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader, _ = load_reader(args.eal2_checkpoint, args.eal2_checkpoint_sha256)
    treatment, treatment_checkpoint = load_pointer(
        args.treatment_checkpoint, args.treatment_checkpoint_sha256
    )
    control, control_checkpoint = load_pointer(
        args.control_checkpoint, args.control_checkpoint_sha256
    )
    reader = reader.to(device).eval()
    treatment = treatment.to(device).eval()
    control = control.to(device).eval()
    reader_hash = module_state_sha256(reader)

    evidence = [item for episode in public for item in episode["evidence"]]
    hidden_evidence = [item for episode in assessor for item in episode["evidence"]]
    temporal_gold = [
        tuple(int(value) // 2 for value in item["numeric_role_ids"])
        for item in hidden_evidence
    ]
    temporal = _predict(
        reader,
        evidence,
        text_key="source_text",
        device=device,
        batch_size=args.batch_size,
    )
    reader_report = _reader_score(evidence, temporal, temporal_gold)
    stride = len(public[0]["evidence"])
    packets: list[EpisodeLawPacket | None] = []
    for index, visible in enumerate(public):
        packets.append(
            compile_episode_laws(
                visible,
                temporal[index * stride : (index + 1) * stride],
                reader_state_sha256=reader_hash,
            ).packet
        )

    normal_records = _command_records(
        public, text_key="command_text", aliases_key="aliases"
    )
    renamed_records = _command_records(
        public, text_key="renamed_command_text", aliases_key="renamed_aliases"
    )
    reverse_records = _command_records(
        public, text_key="reverse_command_text", aliases_key="aliases"
    )
    scrubbed_records = _command_records(
        public, text_key="scrubbed_command_text", aliases_key="aliases"
    )
    predictions = {
        "treatment": _predict_commands(
            treatment, normal_records, device=device, batch_size=args.batch_size
        ),
        "renamed": _predict_commands(
            treatment, renamed_records, device=device, batch_size=args.batch_size
        ),
        "reverse": _predict_commands(
            treatment, reverse_records, device=device, batch_size=args.batch_size
        ),
        "source_scrub": _predict_commands(
            treatment, scrubbed_records, device=device, batch_size=args.batch_size
        ),
        "shuffled_table": _predict_commands(
            treatment,
            normal_records,
            device=device,
            batch_size=args.batch_size,
            rotate_alias_table=True,
        ),
        "shuffled_table_model": _predict_commands(
            control, normal_records, device=device, batch_size=args.batch_size
        ),
    }
    gold = _gold_sequences(assessor)
    reverse_gold = _gold_sequences(assessor, reverse=True)
    program = {
        name: _program_score(values, reverse_gold if name == "reverse" else gold)
        for name, values in predictions.items()
    }
    execution = {
        name: _score_packets(
            packets,
            _candidate_public(public, predictions[name]),
            assessor,
        )
        for name in (
            "treatment",
            "renamed",
            "source_scrub",
            "shuffled_table",
            "shuffled_table_model",
        )
    }

    runtime_path = Path(__file__).with_name("diverge_ncp1_runtime.py")
    runtime_source = runtime_path.read_text()
    source_audit = {
        "runtime": str(runtime_path),
        "runtime_sha256": hashlib.sha256(runtime_source.encode()).hexdigest(),
        "contains_exact_alias_search": any(
            value in runtime_source
            for value in (".find(", "re.search", "aliases.index")
        ),
        "uses_dynamic_alias_pointer": "torch.einsum" in runtime_source,
        "uses_ctc_decode": "greedy_ctc_decode" in runtime_source,
    }
    conditions = {
        "parent_reader_at_least_99_percent": reader_report["complete_exact_rate"]
        >= 0.99,
        "law_commits_at_least_99_percent": sum(packet is not None for packet in packets)
        / len(packets)
        >= 0.99,
        "normal_program_at_least_99_percent": program["treatment"]["exact_rate"]
        >= 0.99,
        "normal_depth_floor_at_least_95_percent": program["treatment"][
            "minimum_depth_rate"
        ]
        >= 0.95,
        "renamed_program_at_least_99_percent": program["renamed"]["exact_rate"] >= 0.99,
        "reverse_program_at_least_99_percent": program["reverse"]["exact_rate"] >= 0.99,
        "normal_state_at_least_99_percent": execution["treatment"]["state_exact_rate"]
        >= 0.99,
        "normal_query_at_least_99_percent": execution["treatment"]["query_exact_rate"]
        >= 0.99,
        "renamed_state_at_least_99_percent": execution["renamed"]["state_exact_rate"]
        >= 0.99,
        "source_scrub_program_at_most_5_percent": program["source_scrub"]["exact_rate"]
        <= 0.05,
        "shuffled_table_program_at_most_5_percent": program["shuffled_table"][
            "exact_rate"
        ]
        <= 0.05,
        "control_model_program_at_most_5_percent": program["shuffled_table_model"][
            "exact_rate"
        ]
        <= 0.05,
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
        "runtime_has_no_exact_alias_search": not source_audit[
            "contains_exact_alias_search"
        ],
        "commands_deleted_before_execution": all(
            "command_text" not in transfer
            for episode in _candidate_public(public, predictions["treatment"])
            for transfer in episode["transfer"]
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
        "law_commits": sum(packet is not None for packet in packets),
        "program": program,
        "execution": execution,
        "source_audit": source_audit,
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "status": report["status"],
                "program_rate": program["treatment"]["exact_rate"],
                "renamed_rate": program["renamed"]["exact_rate"],
                "state_rate": execution["treatment"]["state_exact_rate"],
                "control_rate": program["shuffled_table_model"]["exact_rate"],
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
