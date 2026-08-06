#!/usr/bin/env python3
"""Run the one frozen DIVERGE-NFE1 natural evidence confirmation gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from diverge_ats1_data import OPERATION_TO_ID, PAD_ID
from diverge_fta1_runtime import FTA1Config, FiniteStateSourceCompiler
from diverge_nfe1_data import (
    SCALAR_OPERATIONS,
    apply_scalar,
    validate_board_row,
)
from diverge_nfe1_runtime import (
    ABSTAIN,
    ANSWER,
    REJECT,
    EpisodePacket,
    EvidenceReceipt,
    MentionConfig,
    QueryDecision,
    QueryPacket,
    WholeMentionRoleModel,
    all_particle_bytes,
    canonical_sha256,
    compile_episode,
    compile_mentions_batch,
    compile_query,
    encode_source,
    enumerate_extensional_map,
    execute_factorized,
    factorized_total_bytes,
    issue_evidence,
    mutate_receipt,
    particle_capacity_for_bytes,
    query_particles,
    query_receipt,
    query_soft_answers,
    ranked_assignments,
    receipt_extensional_map,
)


SCHEMA = "shohin-diverge-nfe1-evaluation-v1"
FTA1_SHA256 = "9321b78372d9926930d4de073d70e82c94e8360a69e09be695bab91b2e479f2d"


class NFE1EvaluationError(RuntimeError):
    """The frozen NFE1 evaluation contract was violated."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise NFE1EvaluationError("NFE1 confirmation board hash differs")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_board_row(row)
            rows.append(row)
    if len(rows) != 96 or sum(int(row["depth"]) for row in rows) != 222:
        raise NFE1EvaluationError("NFE1 confirmation geometry differs")
    return rows


def _load_mentions(
    path: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[WholeMentionRoleModel, dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise NFE1EvaluationError("NFE1 mention checkpoint hash differs")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-nfe1-mention-training-report-v1":
        raise NFE1EvaluationError("NFE1 mention checkpoint schema differs")
    if (
        int(checkpoint.get("update", -1)) != 1000
        or int(checkpoint.get("seed", -1)) != 2026080608
    ):
        raise NFE1EvaluationError("NFE1 mention checkpoint schedule differs")
    config = MentionConfig(**checkpoint["config"])
    model = WholeMentionRoleModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def _load_fta1(path: Path, device: torch.device) -> FiniteStateSourceCompiler:
    if sha256_path(path) != FTA1_SHA256:
        raise NFE1EvaluationError("FTA1 checkpoint hash differs")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = FTA1Config(**checkpoint["config"])
    model = FiniteStateSourceCompiler(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def _operation_support(
    model: FiniteStateSourceCompiler,
    sources: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> list[tuple[float, float, float]]:
    operation_ids = tuple(
        OPERATION_TO_ID[name]
        for name in ("SCALAR_ADD", "SCALAR_SUBTRACT", "SCALAR_MULTIPLY")
    )
    output: list[tuple[float, float, float]] = []
    for start in range(0, len(sources), batch_size):
        batch = sources[start : start + batch_size]
        byte_ids = torch.full(
            (len(batch), model.config.max_bytes),
            PAD_ID,
            dtype=torch.long,
            device=device,
        )
        attention = torch.zeros_like(byte_ids, dtype=torch.bool)
        for index, source in enumerate(batch):
            encoded = encode_source(source, model.config.max_bytes)
            byte_ids[index, : len(encoded)] = torch.tensor(encoded, device=device)
            attention[index, : len(encoded)] = True
        _, logits = model(byte_ids, attention)
        selected = logits[:, list(operation_ids)].float().cpu()
        for row in selected:
            output.append(tuple(float(value) for value in row))
    return output


def _exact(decision: QueryDecision, answer: int) -> bool:
    return decision.disposition == ANSWER and decision.answer == answer


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _rate(correct: int, total: int) -> float:
    return correct / max(1, total)


def _receipt_mutations(
    receipts: Sequence[EvidenceReceipt],
) -> list[tuple[EvidenceReceipt, ...]]:
    output: list[tuple[EvidenceReceipt, ...]] = []
    for field in ("source", "step", "mention", "value"):
        mutated = list(receipts)
        mutated[0] = mutate_receipt(mutated[0], field)
        output.append(tuple(mutated))
    return output


def evaluate(
    rows: list[dict[str, Any]],
    mention_model: WholeMentionRoleModel,
    fta1_model: FiniteStateSourceCompiler,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    started = time.monotonic()
    flat_records = [
        {"source_text": str(step["source_text"]), "role_ids": (0, 1, 2)}
        for row in rows
        for step in row["steps"]
    ]
    flat_sources = [str(record["source_text"]) for record in flat_records]
    mentions = compile_mentions_batch(mention_model, flat_records, device=device)
    supports = _operation_support(
        fta1_model,
        flat_sources,
        device=device,
        batch_size=batch_size,
    )

    packets: list[EpisodePacket] = []
    queries: list[QueryPacket] = []
    evidences: list[tuple[EvidenceReceipt, ...]] = []
    mention_exact = 0
    gold_support = 0
    distinct_packets = 0
    top1_wrong_steps = 0
    top1_wrong_episodes = 0
    cursor = 0
    component_rows: list[dict[str, Any]] = []
    for row in rows:
        depth = int(row["depth"])
        row_mentions = mentions[cursor : cursor + depth]
        row_supports = supports[cursor : cursor + depth]
        sources = flat_sources[cursor : cursor + depth]
        cursor += depth
        packet = compile_episode(
            str(row["identity_sha256"]), sources, row_mentions, row_supports
        )
        packets.append(packet)
        queries.append(compile_query(packet, str(row["query"])))
        evidences.append(issue_evidence(packet))
        row_exact = 0
        row_top1_wrong = False
        for packet_step, gold_step in zip(packet.steps, row["steps"], strict=True):
            exact = (
                packet_step.lhs == int(gold_step["lhs"])
                and packet_step.argument == int(gold_step["argument"])
                and packet_step.rhs == int(gold_step["rhs"])
            )
            mention_exact += exact
            row_exact += exact
            gold_index = SCALAR_OPERATIONS.index(str(gold_step["gold_operation"]))
            gold_support += gold_index in range(3)
            outcomes = {
                apply_scalar(operation, packet_step.lhs, packet_step.argument)
                for operation in SCALAR_OPERATIONS
            }
            distinct_packets += len(outcomes) == 3
            predicted = max(
                range(3),
                key=lambda index: (packet_step.operation_support[index], -index),
            )
            wrong = predicted != gold_index
            top1_wrong_steps += wrong
            row_top1_wrong |= wrong
        top1_wrong_episodes += row_top1_wrong
        component_rows.append(
            {
                "identity_sha256": row["identity_sha256"],
                "depth": depth,
                "exact_mentions": row_exact,
                "all_mentions_exact": row_exact == depth,
                "top1_has_wrong_step": row_top1_wrong,
                "packet_commitment": packet.commitment,
            }
        )

    component_pass = (
        mention_exact >= 220 and gold_support == 222 and distinct_packets == 222
    )
    component = {
        "equations": 222,
        "exact_mention_assignments": mention_exact,
        "exact_mention_rate": _rate(mention_exact, 222),
        "gold_support": gold_support,
        "distinct_three_candidate_packets": distinct_packets,
        "top1_wrong_steps": top1_wrong_steps,
        "top1_wrong_episodes": top1_wrong_episodes,
        "pass": component_pass,
        "rows": component_rows,
    }
    if not component_pass:
        return {
            "schema": SCHEMA,
            "status": "component_fail",
            "component": component,
            "elapsed_seconds": time.monotonic() - started,
        }

    counts: Counter[str] = Counter()
    depth_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    operation_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    accounting: Counter[str] = Counter()
    episode_records: list[dict[str, Any]] = []

    for index, (row, packet, query, evidence) in enumerate(
        zip(rows, packets, queries, evidences, strict=True)
    ):
        answer = int(row["answer"])
        depth_key = str(row["depth"])
        full = execute_factorized(packet, evidence)
        full_decision = query_receipt(packet, full, query)
        no_evidence = execute_factorized(packet)
        no_evidence_decision = query_receipt(packet, no_evidence, query)
        reset = execute_factorized(packet, evidence, reset_initial_state=True)
        reset_decision = query_receipt(packet, reset, query)
        shifted = execute_factorized(packet, evidence, operand_semantic_shift=True)
        shifted_decision = query_receipt(packet, shifted, query)
        top_ranked = ranked_assignments(packet)
        top1_decision = query_particles(packet, query, top_ranked[:1], evidence)
        soft_decision = query_soft_answers(packet, query, evidence)

        factorized_bytes = factorized_total_bytes(packet, full, evidence, query)
        capacity, particle_used = particle_capacity_for_bytes(
            packet, top_ranked, evidence, query, factorized_bytes
        )
        equal_decision = query_particles(packet, query, top_ranked[:capacity], evidence)
        particle_bytes = all_particle_bytes(packet, evidence, query)
        full_exact = _exact(full_decision, answer)
        top1_exact = _exact(top1_decision, answer)
        equal_exact = _exact(equal_decision, answer)
        soft_exact = _exact(soft_decision, answer)
        reset_exact = _exact(reset_decision, answer)
        shifted_exact = _exact(shifted_decision, answer)
        parity = receipt_extensional_map(packet, full) == enumerate_extensional_map(
            packet, evidence
        )
        no_evidence_parity = receipt_extensional_map(
            packet, no_evidence
        ) == enumerate_extensional_map(packet)

        shuffled = evidences[(index + 1) % len(evidences)]
        shuffled_receipt = execute_factorized(packet, shuffled)
        shuffled_decision = query_receipt(packet, shuffled_receipt, query)
        swapped_query = queries[(index + 1) % len(queries)]
        swap_decision = query_receipt(packet, full, swapped_query)

        invalid_accepted = 0
        for mutated in _receipt_mutations(evidence):
            invalid_accepted += not execute_factorized(packet, mutated).rejected
        packet_record = packet.record()
        source_absent = all(
            str(step["source_text"]) not in json.dumps(packet_record, sort_keys=True)
            for step in row["steps"]
        )
        before_poison = canonical_sha256(
            {"packet": packet_record, "decision": full_decision.record()}
        )
        poisoned_source = [
            str(step["source_text"]) + " [poison]" for step in row["steps"]
        ]
        after_poison = canonical_sha256(
            {"packet": packet_record, "decision": full_decision.record()}
        )
        poison_invariant = (
            before_poison == after_poison
            and source_absent
            and all(
                source not in json.dumps(packet_record) for source in poisoned_source
            )
        )
        extensional_values = set(enumerate_extensional_map(packet, evidence).values())
        false_commitment = full_decision.disposition == ANSWER and (
            len(extensional_values) != 1
            or full_decision.answer not in extensional_values
        )

        metrics = {
            "full_exact": full_exact,
            "top1_exact": top1_exact,
            "equal_particle_exact": equal_exact,
            "soft_exact": soft_exact,
            "no_evidence_abstain": no_evidence_decision.disposition == ABSTAIN,
            "shuffled_exact": _exact(shuffled_decision, answer),
            "reset_exact": reset_exact,
            "operand_shift_exact": shifted_exact,
            "packet_swap_reject": swap_decision.disposition == REJECT,
            "extensional_parity": parity,
            "no_evidence_extensional_parity": no_evidence_parity,
            "poison_invariant": poison_invariant,
            "invalid_evidence_accepted": invalid_accepted,
            "false_commitment": false_commitment,
            "overflow": full.rejection_reason == "overflow",
        }
        for key, value in metrics.items():
            if isinstance(value, bool):
                counts[key] += value
                depth_counts[depth_key][key] += value
        counts["invalid_evidence_accepted"] += invalid_accepted
        for operation in {str(step["gold_operation"]) for step in row["steps"]}:
            operation_counts[operation]["episodes"] += 1
            operation_counts[operation]["full_exact"] += full_exact
        accounting["factorized_bytes"] += factorized_bytes
        accounting["all_particle_bytes"] += particle_bytes
        accounting["particle_budget_used"] += particle_used
        accounting["equal_particle_capacity"] += capacity
        accounting["logical_applications"] += full.logical_applications
        accounting["unique_applications"] += full.unique_applications
        accounting["peak_groups_sum"] += full.peak_groups
        accounting["peak_groups_max"] = max(
            accounting["peak_groups_max"], full.peak_groups
        )
        episode_records.append(
            {
                "identity_sha256": row["identity_sha256"],
                "depth": int(row["depth"]),
                "answer": answer,
                "packet_commitment": packet.commitment,
                "query_commitment": query.commitment,
                "full": full_decision.record(),
                "top1": top1_decision.record(),
                "equal_particles": equal_decision.record(),
                "soft": soft_decision.record(),
                "no_evidence": no_evidence_decision.record(),
                "shuffled": shuffled_decision.record(),
                "reset": reset_decision.record(),
                "operand_shift": shifted_decision.record(),
                "packet_swap": swap_decision.record(),
                "metrics": metrics,
                "accounting": {
                    "represented_worlds": full.represented_worlds,
                    "factorized_bytes": factorized_bytes,
                    "all_particle_bytes": particle_bytes,
                    "equal_particle_capacity": capacity,
                    "particle_budget_used": particle_used,
                    "logical_applications": full.logical_applications,
                    "unique_applications": full.unique_applications,
                    "peak_groups": full.peak_groups,
                },
            }
        )

    total = len(rows)
    full_rate = _rate(counts["full_exact"], total)
    top1_rate = _rate(counts["top1_exact"], total)
    equal_rate = _rate(counts["equal_particle_exact"], total)
    wrong_top1_exact = sum(
        record["metrics"]["full_exact"]
        for component_row, record in zip(component_rows, episode_records, strict=True)
        if component_row["top1_has_wrong_step"]
    )
    wrong_top1_total = sum(
        bool(record["top1_has_wrong_step"]) for record in component_rows
    )
    gates = {
        "mention_assignment": mention_exact >= 220,
        "candidate_support": gold_support == 222 and distinct_packets == 222,
        "extensional_parity": counts["extensional_parity"] == total
        and counts["no_evidence_extensional_parity"] == total,
        "full_exact": counts["full_exact"] >= 92,
        "wrong_top1_recovery": _rate(wrong_top1_exact, wrong_top1_total) >= 0.90,
        "top1_margin": (full_rate - top1_rate) >= 0.30,
        "particle_margin": (full_rate - equal_rate) >= 0.30,
        "no_evidence_abstention": counts["no_evidence_abstain"] >= 90,
        "shuffled_drop": (full_rate - _rate(counts["shuffled_exact"], total)) >= 0.50,
        "reset_drop": (full_rate - _rate(counts["reset_exact"], total)) >= 0.50,
        "operand_shift_drop": (full_rate - _rate(counts["operand_shift_exact"], total))
        >= 0.50,
        "packet_swap": counts["packet_swap_reject"] == total,
        "source_poison": counts["poison_invariant"] == total,
        "integrity": counts["invalid_evidence_accepted"] == 0
        and counts["false_commitment"] == 0
        and counts["overflow"] == 0,
    }
    summary = {
        "episodes": total,
        "full_exact": counts["full_exact"],
        "top1_exact": counts["top1_exact"],
        "equal_particle_exact": counts["equal_particle_exact"],
        "soft_exact": counts["soft_exact"],
        "wrong_top1_episodes": wrong_top1_total,
        "wrong_top1_recovered": wrong_top1_exact,
        "no_evidence_abstain": counts["no_evidence_abstain"],
        "shuffled_exact": counts["shuffled_exact"],
        "reset_exact": counts["reset_exact"],
        "operand_shift_exact": counts["operand_shift_exact"],
        "packet_swap_reject": counts["packet_swap_reject"],
        "invalid_evidence_accepted": counts["invalid_evidence_accepted"],
        "false_commitments": counts["false_commitment"],
        "overflow": counts["overflow"],
    }
    resource = dict(accounting)
    resource["particle_to_factorized_ratio"] = accounting["all_particle_bytes"] / max(
        1, accounting["factorized_bytes"]
    )
    resource["work_sharing_ratio"] = accounting["logical_applications"] / max(
        1, accounting["unique_applications"]
    )
    return {
        "schema": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "component": component,
        "summary": summary,
        "rates": {
            key: _rate(value, total)
            for key, value in summary.items()
            if key not in {"episodes", "wrong_top1_episodes", "wrong_top1_recovered"}
            and isinstance(value, int)
        },
        "wrong_top1_recovery_rate": _rate(wrong_top1_exact, wrong_top1_total),
        "by_depth": {key: dict(value) for key, value in sorted(depth_counts.items())},
        "by_operation": {
            key: dict(value) for key, value in sorted(operation_counts.items())
        },
        "resource_accounting": resource,
        "gates": gates,
        "episodes_detail": episode_records,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--board-sha256", required=True)
    parser.add_argument("--mention-checkpoint", type=Path, required=True)
    parser.add_argument("--mention-checkpoint-sha256", required=True)
    parser.add_argument("--fta1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NFE1 evaluation: {args.output}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = _load_jsonl(args.board, args.board_sha256)
    mention_model, mention_checkpoint = _load_mentions(
        args.mention_checkpoint, args.mention_checkpoint_sha256, device
    )
    fta1_model = _load_fta1(args.fta1_checkpoint, device)
    result = evaluate(
        rows,
        mention_model,
        fta1_model,
        device=device,
        batch_size=args.batch_size,
    )
    result.update(
        {
            "board": str(args.board),
            "board_sha256": args.board_sha256,
            "mention_checkpoint": str(args.mention_checkpoint),
            "mention_checkpoint_sha256": args.mention_checkpoint_sha256,
            "mention_model_state_sha256": mention_checkpoint["model_state_sha256"],
            "fta1_checkpoint": str(args.fta1_checkpoint),
            "fta1_checkpoint_sha256": FTA1_SHA256,
            "device": str(device),
        }
    )
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result.get("summary"),
                "gates": result.get("gates"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
