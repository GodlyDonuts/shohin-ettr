#!/usr/bin/env python3
"""Frozen development evaluator for DIVERGE-EAL1."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_eal1_data import (
    DEVELOPMENT_EPISODES,
    TRANSFER_DEPTHS,
    validate_episode,
)
from diverge_eal1_runtime import (
    EpisodeLawPacket,
    compile_episode_laws,
    execute_program,
    hard_role_permutation,
    load_reader,
    module_state_sha256,
    rebind_packet,
    sha256_path,
    tensorize_sources,
)


SCHEMA = "shohin-diverge-eal1-development-report-v1"


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("EAL1 evaluation data hash differs")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    text_key: str,
    device: torch.device,
    batch_size: int,
) -> list[tuple[int, int, int, int]]:
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        byte_ids, attention, bounds, _ = tensorize_sources(
            batch,
            device,
            text_key=text_key,
            role_key=None,
        )
        logits = model(byte_ids, attention, bounds)
        output.extend(hard_role_permutation(row) for row in logits)
    return output


def _reader_score(
    records: Sequence[Mapping[str, Any]],
    predicted: Sequence[Sequence[int]],
    gold: Sequence[Sequence[int]],
) -> dict[str, Any]:
    complete = 0
    roles = 0
    renderer: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record, guess, target in zip(records, predicted, gold, strict=True):
        key = ":".join(str(value) for value in record["renderer"])
        exact = tuple(guess) == tuple(target)
        complete += exact
        roles += sum(left == right for left, right in zip(guess, target, strict=True))
        renderer[key][0] += int(exact)
        renderer[key][1] += 1
    per_renderer = {
        key: {"exact": value[0], "total": value[1], "rate": value[0] / value[1]}
        for key, value in sorted(renderer.items())
    }
    return {
        "complete_exact": complete,
        "total": len(records),
        "complete_exact_rate": complete / max(1, len(records)),
        "role_accuracy": roles / max(1, 4 * len(records)),
        "renderer": per_renderer,
        "minimum_renderer_rate": min(
            (value["rate"] for value in per_renderer.values()), default=0.0
        ),
    }


def _score_packets(
    packets: Sequence[EpisodeLawPacket | None],
    public: Sequence[Mapping[str, Any]],
    assessor: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    state_exact = 0
    query_exact = 0
    programs = 0
    queries = 0
    abstained_programs = 0
    by_depth: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for packet, visible, hidden in zip(packets, public, assessor, strict=True):
        for program, target in zip(
            visible["transfer"], hidden["transfer"], strict=True
        ):
            gold = tuple(int(value) for value in target["terminal_state"])
            prediction = None if packet is None else execute_program(packet, program)
            exact = prediction == gold
            state_exact += exact
            programs += 1
            abstained_programs += prediction is None
            depth = int(program["depth"])
            by_depth[depth][0] += int(exact)
            by_depth[depth][1] += 1
            for register in range(2):
                query_exact += (
                    prediction is not None and prediction[register] == gold[register]
                )
                queries += 1
    depth_report = {
        str(depth): {"exact": value[0], "total": value[1], "rate": value[0] / value[1]}
        for depth, value in sorted(by_depth.items())
    }
    if tuple(sorted(by_depth)) != tuple(sorted(TRANSFER_DEPTHS)):
        raise RuntimeError("EAL1 transfer depth board differs")
    return {
        "state_exact": state_exact,
        "programs": programs,
        "state_exact_rate": state_exact / max(1, programs),
        "query_exact": query_exact,
        "queries": queries,
        "query_exact_rate": query_exact / max(1, queries),
        "abstained_programs": abstained_programs,
        "by_depth": depth_report,
        "minimum_depth_rate": min(
            (value["rate"] for value in depth_report.values()), default=0.0
        ),
    }


def _mapped_donor_evidence(
    target: Mapping[str, Any], donor: Mapping[str, Any]
) -> list[dict[str, Any]]:
    donor_aliases = tuple(str(value) for value in donor["aliases"])
    target_aliases = tuple(str(value) for value in target["aliases"])
    output = []
    for record in donor["evidence"]:
        item = dict(record)
        donor_operation = str(record["operation"])
        operation_index = donor_aliases.index(donor_operation)
        target_operation = target_aliases[operation_index]
        for text_key, hash_key in (
            ("source_text", "source_sha256"),
            ("counterfactual_text", "counterfactual_sha256"),
            ("scrubbed_text", "scrubbed_sha256"),
        ):
            text = str(record[text_key]).replace(donor_operation, target_operation)
            item[text_key] = text
            item[hash_key] = hashlib.sha256(text.encode("ascii")).hexdigest()
        item["operation"] = target_operation
        output.append(item)
    return output


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--training-report-sha256", required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--public-data-sha256", required=True)
    parser.add_argument("--assessor-data", type=Path, required=True)
    parser.add_argument("--assessor-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing EAL1 development result")
    if sha256_path(args.training_report) != args.training_report_sha256:
        raise SystemExit("EAL1 training report hash differs")
    training_report = json.loads(args.training_report.read_text())
    if training_report.get("status") != "complete":
        raise SystemExit("EAL1 training did not complete")
    public = _load_jsonl(args.public_data, args.public_data_sha256)
    assessor = _load_jsonl(args.assessor_data, args.assessor_data_sha256)
    if len(public) != DEVELOPMENT_EPISODES or len(assessor) != DEVELOPMENT_EPISODES:
        raise SystemExit("EAL1 development board geometry differs")
    for visible, hidden in zip(public, assessor, strict=True):
        validate_episode(visible, hidden)

    model, checkpoint = load_reader(args.checkpoint, args.checkpoint_sha256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    reader_state_sha256 = module_state_sha256(model)
    records = [item for episode in public for item in episode["evidence"]]
    hidden_evidence = [item for episode in assessor for item in episode["evidence"]]
    normal_gold = [item["numeric_role_ids"] for item in hidden_evidence]
    counterfactual_gold = [item["counterfactual_role_ids"] for item in hidden_evidence]
    normal_predictions = _predict(
        model,
        records,
        text_key="source_text",
        device=device,
        batch_size=args.batch_size,
    )
    counterfactual_predictions = _predict(
        model,
        records,
        text_key="counterfactual_text",
        device=device,
        batch_size=args.batch_size,
    )
    scrubbed_predictions = _predict(
        model,
        records,
        text_key="scrubbed_text",
        device=device,
        batch_size=args.batch_size,
    )
    reader = {
        "normal": _reader_score(records, normal_predictions, normal_gold),
        "counterfactual": _reader_score(
            records, counterfactual_predictions, counterfactual_gold
        ),
        "temporal_scrub": _reader_score(records, scrubbed_predictions, normal_gold),
    }

    stride = len(public[0]["evidence"])
    learned_compilations = []
    oracle_compilations = []
    one_example_compilations = []
    for index, (visible, hidden) in enumerate(zip(public, assessor, strict=True)):
        assignments = normal_predictions[index * stride : (index + 1) * stride]
        oracle_roles = [item["numeric_role_ids"] for item in hidden["evidence"]]
        learned_compilations.append(
            compile_episode_laws(
                visible,
                assignments,
                reader_state_sha256=reader_state_sha256,
            )
        )
        oracle_compilations.append(
            compile_episode_laws(
                visible,
                oracle_roles,
                reader_state_sha256="oracle-role-assessor",
            )
        )
        one_example_compilations.append(
            compile_episode_laws(
                visible,
                oracle_roles,
                reader_state_sha256="oracle-role-assessor",
                evidence_limit_per_operation=1,
            )
        )
    learned_packets = [value.packet for value in learned_compilations]
    oracle_packets = [value.packet for value in oracle_compilations]
    learned_commits = sum(packet is not None for packet in learned_packets)
    oracle_commits = sum(packet is not None for packet in oracle_packets)
    one_example_commits = sum(
        value.packet is not None for value in one_example_compilations
    )
    learned_row_exact = 0
    total_rows = DEVELOPMENT_EPISODES * 8 * 2
    for packet, hidden in zip(learned_packets, assessor, strict=True):
        if packet is None:
            continue
        gold = tuple(
            tuple(tuple(int(coefficient) for coefficient in row) for row in matrix)
            for matrix in hidden["matrices"]
        )
        learned_row_exact += sum(
            learned == expected
            for learned_matrix, gold_matrix in zip(packet.rows, gold, strict=True)
            for learned, expected in zip(learned_matrix, gold_matrix, strict=True)
        )
    execution = {
        "learned": _score_packets(learned_packets, public, assessor),
        "oracle_roles": _score_packets(oracle_packets, public, assessor),
        "reset": _score_packets([None] * len(public), public, assessor),
    }

    transplant_packets = []
    for index, visible in enumerate(public):
        donor = learned_packets[(index + 1) % len(public)]
        transplant_packets.append(
            None if donor is None else rebind_packet(donor, visible["aliases"])
        )
    execution["unrelated_law_transplant"] = _score_packets(
        transplant_packets, public, assessor
    )

    shuffled_public = []
    shuffled_records = []
    for index, visible in enumerate(public):
        donor = public[(index + 1) % len(public)]
        evidence = _mapped_donor_evidence(visible, donor)
        shuffled = {**visible, "evidence": evidence}
        shuffled_public.append(shuffled)
        shuffled_records.extend(evidence)
    shuffled_predictions = _predict(
        model,
        shuffled_records,
        text_key="source_text",
        device=device,
        batch_size=args.batch_size,
    )
    shuffled_packets = []
    for index, visible in enumerate(shuffled_public):
        assignments = shuffled_predictions[index * stride : (index + 1) * stride]
        shuffled_packets.append(
            compile_episode_laws(
                visible,
                assignments,
                reader_state_sha256=reader_state_sha256,
            ).packet
        )
    execution["shuffled_episode_evidence"] = _score_packets(
        shuffled_packets, public, assessor
    )

    packet_source_deleted = all(
        packet is None
        or all(
            text not in json.dumps(packet.record(), sort_keys=True)
            for item in visible["evidence"]
            for text in (
                item["source_text"],
                item["counterfactual_text"],
                item["scrubbed_text"],
            )
        )
        for packet, visible in zip(learned_packets, public, strict=True)
    )
    runtime_path = Path(__file__).with_name("diverge_eal1_runtime.py")
    runtime_source = runtime_path.read_text(encoding="utf-8")
    source_audit = {
        "runtime_path": str(runtime_path),
        "runtime_sha256": hashlib.sha256(runtime_source.encode()).hexdigest(),
        "imports_eal1_data": "from diverge_eal1_data" in runtime_source,
        "imports_exact_pl1_operation": "diverge_pl1_data" in runtime_source,
        "contains_oracle_transition": "oracle_transition" in runtime_source,
        "contains_runtime_verifier": "verifier" in runtime_source.lower(),
        "sealed_packets_exclude_sources": packet_source_deleted,
    }
    conditions = {
        "normal_reader_at_least_99_percent": reader["normal"]["complete_exact_rate"]
        >= 0.99,
        "counterfactual_reader_at_least_99_percent": reader["counterfactual"][
            "complete_exact_rate"
        ]
        >= 0.99,
        "normal_renderer_floor_at_least_95_percent": reader["normal"][
            "minimum_renderer_rate"
        ]
        >= 0.95,
        "counterfactual_renderer_floor_at_least_95_percent": reader["counterfactual"][
            "minimum_renderer_rate"
        ]
        >= 0.95,
        "temporal_scrub_complete_at_most_30_percent": reader["temporal_scrub"][
            "complete_exact_rate"
        ]
        <= 0.30,
        "learned_law_commit_at_least_99_percent": learned_commits / DEVELOPMENT_EPISODES
        >= 0.99,
        "learned_row_exact_at_least_99_percent": learned_row_exact / total_rows >= 0.99,
        "learned_state_exact_at_least_99_percent": execution["learned"][
            "state_exact_rate"
        ]
        >= 0.99,
        "learned_query_exact_at_least_99_percent": execution["learned"][
            "query_exact_rate"
        ]
        >= 0.99,
        "learned_depth_floor_at_least_95_percent": execution["learned"][
            "minimum_depth_rate"
        ]
        >= 0.95,
        "oracle_ceiling_exact": oracle_commits == DEVELOPMENT_EPISODES
        and execution["oracle_roles"]["state_exact_rate"] == 1.0
        and execution["oracle_roles"]["query_exact_rate"] == 1.0,
        "one_example_never_commits": one_example_commits == 0,
        "shuffled_evidence_state_at_most_5_percent": execution[
            "shuffled_episode_evidence"
        ]["state_exact_rate"]
        <= 0.05,
        "shuffled_evidence_query_at_most_5_percent": execution[
            "shuffled_episode_evidence"
        ]["query_exact_rate"]
        <= 0.05,
        "transplant_state_at_most_5_percent": execution["unrelated_law_transplant"][
            "state_exact_rate"
        ]
        <= 0.05,
        "transplant_query_at_most_5_percent": execution["unrelated_law_transplant"][
            "query_exact_rate"
        ]
        <= 0.05,
        "reset_abstains": execution["reset"]["abstained_programs"]
        == execution["reset"]["programs"],
        "candidate_runtime_has_no_exact_semantics_or_verifier": not source_audit[
            "imports_eal1_data"
        ]
        and not source_audit["imports_exact_pl1_operation"]
        and not source_audit["contains_oracle_transition"]
        and not source_audit["contains_runtime_verifier"],
        "sources_deleted_before_execution": packet_source_deleted,
        "training_custody_matches": checkpoint["data_sha256"]
        == training_report.get("data_sha256")
        and checkpoint["source_commit"] == training_report.get("source_commit")
        and checkpoint["model_state_sha256"]
        == training_report.get("final_state_sha256")
        and args.checkpoint_sha256 == training_report.get("checkpoint_sha256"),
    }
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "source_commit": args.source_commit,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "reader_state_sha256": reader_state_sha256,
        "training_report": str(args.training_report),
        "training_report_sha256": args.training_report_sha256,
        "data": {
            "public": str(args.public_data),
            "public_sha256": args.public_data_sha256,
            "assessor": str(args.assessor_data),
            "assessor_sha256": args.assessor_data_sha256,
        },
        "reader": reader,
        "law_compilation": {
            "learned_commits": learned_commits,
            "oracle_commits": oracle_commits,
            "one_example_commits": one_example_commits,
            "episodes": DEVELOPMENT_EPISODES,
            "learned_row_exact": learned_row_exact,
            "total_rows": total_rows,
            "learned_row_exact_rate": learned_row_exact / total_rows,
            "learned_errors": {
                "none" if error is None else error: sum(
                    value.error == error for value in learned_compilations
                )
                for error in (None, "empty_support", "underdetermined")
            },
        },
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
                "normal_reader": reader["normal"]["complete_exact_rate"],
                "counterfactual_reader": reader["counterfactual"][
                    "complete_exact_rate"
                ],
                "scrub": reader["temporal_scrub"]["complete_exact_rate"],
                "learned_state": execution["learned"]["state_exact_rate"],
                "learned_query": execution["learned"]["query_exact_rate"],
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
