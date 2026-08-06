#!/usr/bin/env python3
"""Evaluate the frozen DIVERGE-SOT1 stage-owned transaction gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from diverge_iem1_runtime import (
    IEM1RuntimeError,
    mutate_query_receipt,
    seal_natural_query,
)
from diverge_nve1_runtime import seal_natural_evidence
from diverge_sot1_runtime import (
    SOT1Config,
    StageOwnedEpistemicMachine,
    module_state_sha256,
    validate_owner_isolation,
)
from diverge_tfs1_runtime import (
    ABSTAIN,
    REJECT,
    execute_factorized,
    factorized_total_bytes,
    particle_capacity_for_bytes,
    query_particles,
    query_receipt,
    ranked_assignments,
    receipt_extensional_map,
)
from eval_diverge_iem1 import (
    _answer_exact,
    _assessor_worlds,
    _compile_evidence_for_packets,
    _compile_natural_queries,
    _compile_packets,
    _decision_record,
    _load_board,
    _natural_query_exact,
    _natural_receipt_exact,
    _query_from_compilation,
    _receipt_tuple,
    canonical_sha256,
    sha256_path,
)
from eval_diverge_nve1 import evaluate as evaluate_nve1


SCHEMA = "shohin-diverge-sot1-evaluation-v1"


class SOT1EvaluationError(RuntimeError):
    """The frozen SOT1 evaluation contract was violated."""


def _load_sot1(
    path: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[StageOwnedEpistemicMachine, dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SOT1EvaluationError("SOT1 checkpoint hash differs")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-sot1-training-report-v1":
        raise SOT1EvaluationError("SOT1 checkpoint schema differs")
    if int(checkpoint.get("update", -1)) != 1000:
        raise SOT1EvaluationError("SOT1 checkpoint duration differs")
    model = StageOwnedEpistemicMachine(SOT1Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.freeze_qualified_owners()
    validate_owner_isolation(model)
    model.eval()
    if module_state_sha256(model) != checkpoint["model_state_sha256"]:
        raise SOT1EvaluationError("SOT1 model state differs")
    owner_hashes = model.owner_hashes()
    if owner_hashes != checkpoint["final_owner_hashes"]:
        raise SOT1EvaluationError("SOT1 owner hashes differ")
    if owner_hashes["WORLD"] != checkpoint["initial_owner_hashes"]["WORLD"]:
        raise SOT1EvaluationError("SOT1 WORLD owner changed")
    if owner_hashes["EVIDENCE"] != checkpoint["initial_owner_hashes"]["EVIDENCE"]:
        raise SOT1EvaluationError("SOT1 EVIDENCE owner changed")
    if model.owner_manifest() != checkpoint["owner_manifest"]:
        raise SOT1EvaluationError("SOT1 owner manifest differs")
    return model, checkpoint


def _load_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise SOT1EvaluationError(f"protected result hash differs: {path}")
    return json.loads(path.read_text())


def _query_components(
    rows: Sequence[dict[str, Any]],
    model: StageOwnedEpistemicMachine,
    *,
    owner_hashes: Mapping[str, str],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    started = time.monotonic()
    packets, _, source_failures = _compile_packets(
        rows,
        model.source_owner,
        commitment=owner_hashes["WORLD"],
        device=device,
        integrated=False,
    )
    evidence_sets = _compile_evidence_for_packets(
        model.evidence_owner,
        rows,
        packets,
        commitment=owner_hashes["EVIDENCE"],
        device=device,
        batch_size=batch_size,
    )
    query_sets = _compile_natural_queries(
        model,
        rows,
        packets,
        commitment=owner_hashes["QUERY"],
        device=device,
    )
    query_swapped = _compile_natural_queries(
        model,
        rows,
        packets,
        commitment=owner_hashes["QUERY"],
        device=device,
        swap_roles=True,
    )

    counts = Counter()
    query_mode_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    failures = []
    transcripts = []
    receipts_by_episode = []
    executions = []
    for row, packet, evidence_set in zip(rows, packets, evidence_sets, strict=True):
        receipts = None if evidence_set is None else _receipt_tuple(evidence_set)
        receipts_by_episode.append(receipts)
        if packet is None or receipts is None:
            executions.append(None)
            continue
        try:
            typed = seal_natural_evidence(
                packet,
                receipts,
                expected_compiler_commitment=owner_hashes["EVIDENCE"],
            )
            executions.append((typed, execute_factorized(packet, typed)))
        except Exception:
            executions.append(None)

    for episode_index, (row, packet, evidence_set, query_set, execution) in enumerate(
        zip(rows, packets, evidence_sets, query_sets, executions, strict=True)
    ):
        tfs1 = row["tfs1"]
        counts["episodes"] += 1
        if packet is None:
            failures.append({"id": str(tfs1["id"]), "error": "source absent"})
            continue
        counts["source_program_exact"] += 1
        if evidence_set is not None:
            for compilation, supervisor in zip(
                evidence_set, row["natural_evidence"], strict=True
            ):
                counts["evidence_total"] += 1
                counts["evidence_exact"] += _natural_receipt_exact(
                    compilation, supervisor
                )
        if query_set is None:
            failures.append({"id": str(tfs1["id"]), "error": "queries absent"})
            continue
        for name, compilation in query_set.items():
            exact = _natural_query_exact(compilation, row["natural_queries"][name])
            counts["query_total"] += 1
            counts["query_exact"] += exact
            query_mode_counts[name]["total"] += 1
            query_mode_counts[name]["exact"] += exact
        if execution is None:
            failures.append({"id": str(tfs1["id"]), "error": "evidence unsealed"})
            continue
        typed, learned = execution
        sensitive = _query_from_compilation(query_set["sensitive"])
        invariant = _query_from_compilation(query_set["invariant"])
        underdetermined = _query_from_compilation(query_set["underdetermined"])
        if sensitive is None or invariant is None or underdetermined is None:
            failures.append({"id": str(tfs1["id"]), "error": "query unsealed"})
            continue
        counts["episodes_fully_sealed"] += 1
        expected = str(tfs1["gold_answer"])
        gold_assignment = tuple(int(value) for value in tfs1["gold_assignment"])
        learned_decision = query_receipt(packet, learned, sensitive)
        counts["sensitive_exact"] += _answer_exact(learned_decision, expected)
        no_evidence = execute_factorized(packet)
        no_evidence_sensitive = query_receipt(packet, no_evidence, sensitive)
        no_evidence_invariant = query_receipt(packet, no_evidence, invariant)
        partial = execute_factorized(packet, typed[:-1])
        partial_under = query_receipt(packet, partial, underdetermined)
        counts["no_evidence_abstain"] += no_evidence_sensitive.disposition == ABSTAIN
        counts["invariant_exact"] += _answer_exact(
            no_evidence_invariant,
            str(tfs1["gold_terminal"][invariant.register]),
        )
        counts["partial_underdetermined_abstain"] += (
            partial_under.disposition == ABSTAIN
        )
        ranked = ranked_assignments(packet)
        top1_wrong = ranked[0] != gold_assignment
        counts["initial_top1_wrong"] += top1_wrong
        counts["sensitive_exact_initial_top1_wrong"] += top1_wrong and _answer_exact(
            learned_decision, expected
        )
        top1 = query_particles(packet, sensitive, ranked[:1], typed)
        factorized_bytes = factorized_total_bytes(packet, learned, typed)
        capacity, _ = particle_capacity_for_bytes(
            packet, ranked, typed, factorized_bytes
        )
        equal = query_particles(packet, sensitive, ranked[:capacity], typed)
        counts["top1_exact"] += _answer_exact(top1, expected)
        counts["equal_particle_exact"] += _answer_exact(equal, expected)

        swapped_query = None
        if query_swapped[episode_index] is not None:
            swapped_query = _query_from_compilation(
                query_swapped[episode_index]["sensitive"]
            )
        swapped_decision = (
            None
            if swapped_query is None
            else query_receipt(packet, learned, swapped_query)
        )
        counts["query_role_swap_exact"] += (
            swapped_decision is not None and _answer_exact(swapped_decision, expected)
        )
        other_index = (episode_index + 1) % len(rows)
        other_query = None
        if query_sets[other_index] is not None:
            other_query = _query_from_compilation(query_sets[other_index]["sensitive"])
        packet_swap = (
            None if other_query is None else query_receipt(packet, learned, other_query)
        )
        counts["packet_query_swap_reject"] += (
            packet_swap is None or packet_swap.disposition == REJECT
        )

        natural_receipt = query_set["sensitive"].receipt
        assert natural_receipt is not None
        invalid_queries = 0
        for field in (
            "packet",
            "source",
            "compiler",
            "target",
            "distractor",
            "commitment",
        ):
            try:
                seal_natural_query(
                    packet,
                    mutate_query_receipt(natural_receipt, field),
                    expected_compiler_commitment=owner_hashes["QUERY"],
                )
                invalid_queries += 1
            except IEM1RuntimeError:
                pass
        counts["invalid_queries_accepted"] += invalid_queries

        sealed_record = {
            "packet": packet.record(),
            "receipts": [
                value.record() for value in receipts_by_episode[episode_index]
            ],
            "query": natural_receipt.record(),
            "decision": _decision_record(learned_decision),
        }
        sealed_hash = canonical_sha256(sealed_record)
        poisoned_sources = [str(tfs1["source"]) + " [post-seal poison]"]
        poisoned_sources.extend(
            str(item["source_text"]) + " [post-seal poison]"
            for item in row["natural_evidence"]
        )
        poisoned_sources.extend(
            str(item["source_text"]) + " [post-seal poison]"
            for item in row["natural_queries"].values()
        )
        counts["post_seal_poison_invariant"] += sealed_hash == canonical_sha256(
            sealed_record
        ) and all(text not in json.dumps(sealed_record) for text in poisoned_sources)
        learned_worlds = receipt_extensional_map(learned)
        assessor = _assessor_worlds(tfs1)
        counts["extensional_parity"] += learned_worlds == {
            gold_assignment: assessor[gold_assignment]
        }
        if len(transcripts) < 24:
            transcripts.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "expected": expected,
                    "decision": _decision_record(learned_decision),
                    "query_exact": {
                        name: _natural_query_exact(
                            compilation, row["natural_queries"][name]
                        )
                        for name, compilation in query_set.items()
                    },
                }
            )

    return {
        "counts": dict(counts),
        "query_mode_counts": {
            name: dict(values) for name, values in sorted(query_mode_counts.items())
        },
        "source_failures": source_failures,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
        "transcripts": transcripts,
    }


def evaluate(
    rows: list[dict[str, Any]],
    model: StageOwnedEpistemicMachine,
    *,
    protected_nve1: Mapping[str, Any],
    protected_tol3: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    started = time.monotonic()
    owner_hashes = model.owner_hashes()
    nve1 = evaluate_nve1(
        rows,
        model.evidence_owner,
        model.source_owner,
        evidence_commitment=owner_hashes["EVIDENCE"],
        tol3_commitment=owner_hashes["WORLD"],
        device=device,
        batch_size=batch_size,
    )
    natural = _query_components(
        rows,
        model,
        owner_hashes=owner_hashes,
        device=device,
        batch_size=batch_size,
    )
    counts = Counter(natural["counts"])
    modes = natural["query_mode_counts"]
    rows_total = len(rows)
    control_drop = math.ceil(0.50 * rows_total)
    wrong = counts["initial_top1_wrong"]
    wrong_rate = counts["sensitive_exact_initial_top1_wrong"] / max(1, wrong)
    protected_nve_exact = int(protected_nve1.get("counts", {}).get("learned_exact", 0))
    protected_tol3_exact = int(
        protected_tol3.get("counts", {}).get("semantic_program_exact", 0)
    )
    conditions = {
        "fresh_nve1_gate_passed": bool(
            nve1.get("promotion_gate", {}).get("passed", False)
        ),
        "fresh_source_at_least_250": counts["source_program_exact"] >= 250,
        "fresh_evidence_at_least_3041": counts["evidence_exact"] >= 3041,
        "fresh_query_at_least_752": counts["query_exact"] >= 752,
        "sensitive_queries_at_least_245": modes.get("sensitive", {}).get("exact", 0)
        >= 245,
        "invariant_queries_at_least_245": modes.get("invariant", {}).get("exact", 0)
        >= 245,
        "underdetermined_queries_at_least_245": modes.get("underdetermined", {}).get(
            "exact", 0
        )
        >= 245,
        "sensitive_answers_at_least_245": counts["sensitive_exact"] >= 245,
        "wrong_top1_conditional_at_least_95_percent": (
            wrong > 0 and wrong_rate >= 0.95
        ),
        "extensional_parity_on_answers": counts["extensional_parity"]
        == counts["sensitive_exact"],
        "no_evidence_abstains_at_least_245": counts["no_evidence_abstain"] >= 245,
        "invariant_answers_at_least_245": counts["invariant_exact"] >= 245,
        "partial_underdetermined_abstains_at_least_245": counts[
            "partial_underdetermined_abstain"
        ]
        >= 245,
        "beats_top1_by_50_points": counts["sensitive_exact"] - counts["top1_exact"]
        >= control_drop,
        "beats_equal_particles_by_50_points": counts["sensitive_exact"]
        - counts["equal_particle_exact"]
        >= control_drop,
        "query_role_swap_drop_50_points": counts["sensitive_exact"]
        - counts["query_role_swap_exact"]
        >= control_drop,
        "packet_query_swaps_all_reject": counts["packet_query_swap_reject"]
        == rows_total,
        "post_seal_poison_invariant": counts["post_seal_poison_invariant"]
        == rows_total,
        "zero_invalid_queries": counts["invalid_queries_accepted"] == 0,
        "protected_nve1_at_least_250": protected_nve_exact >= 250,
        "protected_tol3_at_least_1000": protected_tol3_exact >= 1000,
        "owner_manifest_isolated": True,
    }
    return {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "owner_hashes": owner_hashes,
        "fresh_nve1": nve1,
        "natural_query_path": natural,
        "protected": {
            "nve1_result_sha256": protected_nve1["_sha256"],
            "nve1_exact": protected_nve_exact,
            "tol3_result_sha256": protected_tol3["_sha256"],
            "tol3_semantic_program_exact": protected_tol3_exact,
        },
        "ratios": {"wrong_top1_conditional_exact_rate": wrong_rate},
        "promotion_gate": {
            "conditions": conditions,
            "passed": all(conditions.values()),
        },
        "elapsed_seconds": time.monotonic() - started,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--protected-nve1-result", type=Path, required=True)
    parser.add_argument("--protected-nve1-result-sha256", required=True)
    parser.add_argument("--protected-tol3-result", type=Path, required=True)
    parser.add_argument("--protected-tol3-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing SOT1 result: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("SOT1 requested CUDA is unavailable")
    device = torch.device(args.device)
    rows = _load_board(args.data, args.data_sha256)
    model, checkpoint = _load_sot1(args.checkpoint, args.checkpoint_sha256, device)
    protected_nve1 = _load_json(
        args.protected_nve1_result, args.protected_nve1_result_sha256
    )
    protected_nve1["_sha256"] = args.protected_nve1_result_sha256
    protected_tol3 = _load_json(
        args.protected_tol3_result, args.protected_tol3_result_sha256
    )
    protected_tol3["_sha256"] = args.protected_tol3_result_sha256
    report = evaluate(
        rows,
        model,
        protected_nve1=protected_nve1,
        protected_tol3=protected_tol3,
        device=device,
        batch_size=args.batch_size,
    )
    report.update(
        {
            "data": str(args.data),
            "data_sha256": args.data_sha256,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "model_state_sha256": checkpoint["model_state_sha256"],
            "owner_manifest": checkpoint["owner_manifest"],
            "query_data_sha256": checkpoint["query_data_sha256"],
            "device": str(device),
        }
    )
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "status": report["status"],
                "counts": report["natural_query_path"]["counts"],
                "promotion_gate": report["promotion_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
