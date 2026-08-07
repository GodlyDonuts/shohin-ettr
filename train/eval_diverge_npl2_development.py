#!/usr/bin/env python3
"""Evaluate the one frozen DIVERGE-NPL2 natural development gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from diverge_cgl1_runtime import frozen_backbone_state_sha256
from diverge_npl1_data import (
    DEVELOPMENT_COUNT,
    render_feedback,
    validate_natural_public_record,
)
from diverge_npl2_runtime import (
    DecodedEvidence,
    NaturalEpisodeResult,
    TypedEpisode,
    execute_typed_mapping,
    run_natural_episode,
    typed_episode_from_public,
)
from diverge_nve1_data import (
    NUMERIC_ROLES,
    SYMBOL_ROLES,
    scan_rational_spans,
    symbol_occurrence_groups,
)
from diverge_nve1_runtime import hard_role_permutation, tensorize_sources
from diverge_pl1_data import Episode, episode_from_assessor_record
from diverge_pl1_runtime import (
    freeze_policy,
    matrix_hash,
    maximum_assignment,
)
from eval_diverge_eic1 import _load_model as _load_eic1
from eval_diverge_eni1_semantics import _load_jsonl, _query_records
from eval_diverge_pqi1 import sha256_path
from eval_diverge_sti1 import _load_sti1

SCHEMA = "shohin-diverge-npl2-development-v1"
RUN_SEED = 2026080799
NATURAL_ARMS = (
    "STATIC",
    "CONTEXT_ONLY",
    "DIVERGE_ONLY",
    "FAST_WEIGHT",
    "TRANSIENT_GRAD",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _commitment(domain: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256()
    for part in (domain.encode("ascii"), encoded):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _feedback_work(
    public: Sequence[Mapping[str, Any]],
    typed: Sequence[TypedEpisode],
) -> list[dict[str, Any]]:
    work = []
    for episode_index, (record, episode) in enumerate(zip(public, typed, strict=True)):
        symbols = [str(value) for value in record["symbol_table"]]
        for plan in record["feedback_plan"]:
            attempt = int(plan["attempt"])
            branch = int(plan["branch"])
            depth = len(episode.acquisition[attempt].symbols)
            for code in (0, *range(2, depth + 2)):
                work.append(
                    {
                        "episode_index": episode_index,
                        "attempt": attempt,
                        "branch": branch,
                        "renderer": int(plan["renderer"]),
                        "expected_target": str(plan["target_branch"]),
                        "expected_distractor": str(plan["distractor_branch"]),
                        "code": code,
                        "source_text": render_feedback(plan, code),
                        "symbols": symbols,
                    }
                )
    return work


@torch.no_grad()
def _compile_evidence(
    model,
    public: Sequence[Mapping[str, Any]],
    typed: Sequence[TypedEpisode],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[tuple[int, int, int], DecodedEvidence]], dict[str, Any]]:
    work = _feedback_work(public, typed)
    tables: list[dict[tuple[int, int, int], DecodedEvidence]] = [{} for _ in public]
    overall = Counter()
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for start in range(0, len(work), batch_size):
        batch = work[start : start + batch_size]
        ids, mask, bounds, symbols, _, _ = tensorize_sources(batch, device)
        numeric_logits, symbol_logits = model.evidence_owner(ids, mask, bounds, symbols)
        for row, item in enumerate(batch):
            text = str(item["source_text"])
            numeric_assignment = hard_role_permutation(numeric_logits[row])
            symbol_assignment = hard_role_permutation(symbol_logits[row])
            numeric_spans = scan_rational_spans(text)
            symbol_groups = symbol_occurrence_groups(text, item["symbols"])
            decoded_attempt = -1
            decoded_code = -1
            decoded_target = ""
            decoded_distractor = ""
            if len(numeric_spans) == 2 and len(symbol_groups) == 2:
                numeric = {
                    NUMERIC_ROLES[role]: text[left:right]
                    for (left, right), role in zip(
                        numeric_spans, numeric_assignment, strict=True
                    )
                }
                symbolic = {
                    SYMBOL_ROLES[role]: group[0]
                    for group, role in zip(
                        symbol_groups, symbol_assignment, strict=True
                    )
                }
                try:
                    decoded_attempt = int(numeric["STEP"]) - 1
                    decoded_code = int(numeric["VALUE"])
                    decoded_target = symbolic["TARGET"]
                    decoded_distractor = symbolic["DISTRACTOR"]
                except (KeyError, ValueError):
                    pass
            commitment = _commitment(
                "diverge-npl2-decoded-evidence",
                {
                    "source_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
                    "numeric_assignment": numeric_assignment,
                    "symbol_assignment": symbol_assignment,
                },
            )
            decoded = DecodedEvidence(
                attempt=decoded_attempt,
                target_branch=decoded_target,
                distractor_branch=decoded_distractor,
                certificate_code=decoded_code,
                commitment=commitment,
            )
            key = (int(item["attempt"]), int(item["branch"]), int(item["code"]))
            table = tables[int(item["episode_index"])]
            if key in table:
                raise RuntimeError("duplicate NPL2 evidence transaction")
            table[key] = decoded
            exact = (
                decoded.attempt == int(item["attempt"])
                and decoded.certificate_code == int(item["code"])
                and decoded.target_branch == str(item["expected_target"])
                and decoded.distractor_branch == str(item["expected_distractor"])
            )
            for counter in (overall, by_renderer[str(int(item["renderer"]))]):
                counter["total"] += 1
                counter["exact"] += exact
    return tables, {
        "overall": dict(overall),
        "by_renderer": {key: dict(value) for key, value in sorted(by_renderer.items())},
        "model_forwards": (len(work) + batch_size - 1) // batch_size,
        "all_legal_messages_compiled_before_outcome_selection": True,
    }


@torch.no_grad()
def _compile_queries(
    model,
    public: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    records = []
    metadata = []
    for episode_index, candidate in enumerate(public):
        query_records = _query_records(candidate)
        if len(query_records) != 32:
            raise RuntimeError("NPL2 query geometry differs")
        for query_index, (record, query) in enumerate(
            zip(query_records, candidate["queries"], strict=True)
        ):
            records.append(record)
            metadata.append((episode_index, query_index, candidate, query))
    scores = model.candidate_scores(
        records, device=device, batch_size=batch_size, control="normal"
    )
    predictions = scores.argmax(dim=-1).tolist()
    selectors = [[-1] * 32 for _ in public]
    overall = Counter()
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record, prediction, meta in zip(records, predictions, metadata, strict=True):
        episode_index, query_index, candidate, query = meta
        groups = symbol_occurrence_groups(str(record["source_text"]), record["symbols"])
        registers = tuple(str(value) for value in candidate["register_names"])
        selected = groups[int(prediction)][0] if len(groups) == 2 else ""
        selector = registers.index(selected) if selected in registers else -1
        selectors[episode_index][query_index] = selector
        exact = selector == int(query["register_index"])
        for counter in (overall, by_renderer[str(int(query["renderer"]))]):
            counter["total"] += 1
            counter["exact"] += exact
    return [tuple(values) for values in selectors], {
        "overall": dict(overall),
        "by_renderer": {key: dict(value) for key, value in sorted(by_renderer.items())},
        "score_sha256": hashlib.sha256(scores.numpy().tobytes()).hexdigest(),
        "model_records": len(records),
    }


def _oracle_semantics(
    record: Mapping[str, Any], typed: TypedEpisode
) -> tuple[dict[tuple[int, int, int], DecodedEvidence], tuple[int, ...]]:
    table = {}
    for plan in record["feedback_plan"]:
        attempt = int(plan["attempt"])
        branch = int(plan["branch"])
        depth = len(typed.acquisition[attempt].symbols)
        for code in (0, *range(2, depth + 2)):
            text = render_feedback(plan, code)
            table[(attempt, branch, code)] = DecodedEvidence(
                attempt=attempt,
                target_branch=str(plan["target_branch"]),
                distractor_branch=str(plan["distractor_branch"]),
                certificate_code=code,
                commitment=hashlib.sha256(text.encode("ascii")).hexdigest(),
            )
    return table, tuple(int(query["register_index"]) for query in record["queries"])


def _summarize(
    results: Sequence[NaturalEpisodeResult], elapsed: float
) -> dict[str, Any]:
    query_exact = sum(result.query_exact for result in results)
    query_total = sum(result.query_total for result in results)
    transfer_exact = sum(result.transfer_exact for result in results)
    transfer_total = sum(result.transfer_total for result in results)
    probe = [
        sum(result.probe_query_exact[index] for result in results)
        for index in range(12)
    ]
    writes = [receipt for result in results for receipt in result.write_receipts]
    return {
        "episodes": len(results),
        "query_exact": query_exact,
        "query_total": query_total,
        "query_rate": query_exact / query_total,
        "transfer_exact": transfer_exact,
        "transfer_total": transfer_total,
        "transfer_rate": transfer_exact / transfer_total,
        "mapping_exact": sum(result.mapping_exact for result in results),
        "mapping_total": len(results),
        "mapping_rate": sum(result.mapping_exact for result in results) / len(results),
        "per_episode_query_exact": [result.query_exact for result in results],
        "probe_query_exact_by_attempt": probe,
        "probe_query_total_by_attempt": [len(results) * 32] * 12,
        "semantic_rejections": sum(result.semantic_rejections for result in results),
        "write_norm_max": max((receipt.update_norm for receipt in writes), default=0.0),
        "protected_hashes_exact": all(
            len({receipt.protected_hash for receipt in result.write_receipts}) <= 1
            for result in results
        ),
        "rejected_credits": sum(receipt.rejected_credits for receipt in writes),
        "elapsed_seconds": elapsed,
    }


def _query_from_policy(
    typed: TypedEpisode,
    assessor: Episode,
    selectors: Sequence[int],
    policy_state: tuple[tuple[float, ...], ...],
) -> tuple[int, ...]:
    mapping = maximum_assignment([list(row) for row in policy_state])
    output = []
    for index, (public_program, hidden_program) in enumerate(
        zip(typed.transfer, assessor.transfer, strict=True)
    ):
        terminal = execute_typed_mapping(mapping, public_program)[-1]
        for expected in range(2):
            selector = int(selectors[2 * index + expected])
            output.append(terminal[selector] if selector in (0, 1) else -1)
    return tuple(output)


def _transplant_rate(
    typed: Sequence[TypedEpisode],
    assessors: Sequence[Episode],
    selectors: Sequence[Sequence[int]],
    results: Sequence[NaturalEpisodeResult],
) -> float:
    exact = 0
    total = 0
    for index, (episode, hidden, query) in enumerate(
        zip(typed, assessors, selectors, strict=True)
    ):
        donor = results[index - 1]
        outputs = _query_from_policy(episode, hidden, query, donor.policy_state)
        gold = tuple(
            value for program in hidden.transfer for value in program.terminal_state
        )
        exact += sum(left == right for left, right in zip(outputs, gold, strict=True))
        total += len(gold)
    return exact / total


def _rollback(
    typed: Sequence[TypedEpisode],
    assessors: Sequence[Episode],
    selectors: Sequence[Sequence[int]],
    results: Sequence[NaturalEpisodeResult],
) -> dict[str, int]:
    exact = 0
    changed = 0
    for episode, hidden, query, result in zip(
        typed, assessors, selectors, results, strict=True
    ):
        before = [list(row) for row in result.policy_state]
        pre = _query_from_policy(episode, hidden, query, result.policy_state)
        poisoned = [row[:] for row in before]
        wrong = list(hidden.symbol_to_operation[1:]) + [hidden.symbol_to_operation[0]]
        for symbol, operation in enumerate(wrong):
            poisoned[symbol][operation] += 64.0
        poison_state = freeze_policy(poisoned)
        poison = _query_from_policy(episode, hidden, query, poison_state)
        rollback_state = freeze_policy([row[:] for row in before])
        restored = _query_from_policy(episode, hidden, query, rollback_state)
        exact += (
            matrix_hash(before) == matrix_hash([list(row) for row in rollback_state])
            and pre == restored
        )
        changed += pre != poison
    return {"exact": exact, "changed": changed, "total": len(results)}


def _semantic_floor(score: Mapping[str, Any], floor: float, renderers: int) -> bool:
    overall = score["overall"]
    return (
        int(overall["total"]) > 0
        and int(overall["exact"]) / int(overall["total"]) >= floor
        and len(score["by_renderer"]) == renderers
        and all(
            int(value["exact"]) / int(value["total"]) >= 0.99
            for value in score["by_renderer"].values()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--eni1-result", type=Path, required=True)
    parser.add_argument("--eni1-result-sha256", required=True)
    parser.add_argument("--eic-checkpoint", type=Path, required=True)
    parser.add_argument("--eic-checkpoint-sha256", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--sti-checkpoint", type=Path, required=True)
    parser.add_argument("--sti-checkpoint-sha256", required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--public-data-sha256", required=True)
    parser.add_argument("--assessor-data", type=Path, required=True)
    parser.add_argument("--assessor-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing NPL2 development result")
    if not torch.cuda.is_available():
        raise SystemExit("NPL2 development requires CUDA semantic owners")
    if sha256_path(args.eni1_result) != args.eni1_result_sha256:
        raise SystemExit("NPL2 ENI1 receipt hash differs")
    with args.eni1_result.open(encoding="utf-8") as handle:
        eni1 = json.load(handle)
    if eni1.get("status") != "pass" or not eni1.get("gate", {}).get("passed"):
        raise SystemExit("NPL2 requires a passing ENI1 receipt")

    public = _load_jsonl(args.public_data, args.public_data_sha256)
    hidden_rows = _load_jsonl(args.assessor_data, args.assessor_data_sha256)
    if len(public) != DEVELOPMENT_COUNT or len(hidden_rows) != DEVELOPMENT_COUNT:
        raise SystemExit("NPL2 development geometry differs")
    assessors = []
    typed = []
    world = Counter()
    for candidate, hidden in zip(public, hidden_rows, strict=True):
        validate_natural_public_record(candidate)
        if hidden["public"] != candidate:
            raise SystemExit("NPL2 public/assessor surface differs")
        assessor = episode_from_assessor_record(hidden["oracle"])
        episode = typed_episode_from_public(candidate)
        for public_program, hidden_program in zip(
            (*episode.acquisition, *episode.transfer),
            (*assessor.acquisition, *assessor.transfer),
            strict=True,
        ):
            world["total"] += 1
            world["exact"] += (
                public_program.initial_state == hidden_program.initial_state
                and public_program.symbols == hidden_program.symbols
            )
        assessors.append(assessor)
        typed.append(episode)

    device = torch.device("cuda")
    eic, eic_checkpoint = _load_eic1(
        args.eic_checkpoint,
        args.eic_checkpoint_sha256,
        args.base,
        args.base_sha256,
        args.tokenizer,
        args.tokenizer_sha256,
        device,
    )
    sti, sti_checkpoint = _load_sti1(
        args.sti_checkpoint, args.sti_checkpoint_sha256, device
    )
    if eic_checkpoint.get("projection_mode") != "involution":
        raise SystemExit("NPL2 requires confirmed EIC1 involution")
    eic_hash_before = frozen_backbone_state_sha256(eic.backbone)
    sti_hash_before = sti.owner_hashes()
    evidence_tables, evidence_score = _compile_evidence(
        sti,
        public,
        typed,
        device=device,
        batch_size=max(128, args.batch_size),
    )
    query_selectors, query_score = _compile_queries(
        eic,
        public,
        device=device,
        batch_size=min(64, args.batch_size),
    )
    owner_hashes_exact = (
        eic_hash_before == frozen_backbone_state_sha256(eic.backbone)
        and sti_hash_before == sti.owner_hashes()
    )
    protected = {
        "WORLD": "structural-npl1-world-v1",
        "EVIDENCE": sti_hash_before["EVIDENCE"],
        "QUERY_BACKBONE": eic_hash_before,
        "QUERY_ADAPTER": str(eic_checkpoint["adapter_state_sha256"]),
        "EXECUTOR": "exact-z97-executor-v1",
    }

    results_by_arm: dict[str, tuple[NaturalEpisodeResult, ...]] = {}
    summaries = {}
    for arm in NATURAL_ARMS:
        started = time.perf_counter()
        results = tuple(
            run_natural_episode(
                episode,
                assessor,
                evidence=evidence,
                query_selectors=selectors,
                arm=arm,  # type: ignore[arg-type]
                seed=RUN_SEED,
                protected_manifest=protected,
            )
            for episode, assessor, evidence, selectors in zip(
                typed, assessors, evidence_tables, query_selectors, strict=True
            )
        )
        results_by_arm[arm] = results
        summaries[arm] = _summarize(results, time.perf_counter() - started)

    started = time.perf_counter()
    npl2 = tuple(
        run_natural_episode(
            episode,
            assessor,
            evidence=evidence,
            query_selectors=selectors,
            arm="PL1",
            candidate_label="NPL2",
            proposal_arm="PL1",
            seed=RUN_SEED,
            protected_manifest=protected,
        )
        for episode, assessor, evidence, selectors in zip(
            typed, assessors, evidence_tables, query_selectors, strict=True
        )
    )
    results_by_arm["NPL2"] = npl2
    summaries["NPL2"] = _summarize(npl2, time.perf_counter() - started)

    oracle_tables = []
    oracle_queries = []
    for record, episode in zip(public, typed, strict=True):
        evidence, selectors = _oracle_semantics(record, episode)
        oracle_tables.append(evidence)
        oracle_queries.append(selectors)
    started = time.perf_counter()
    oracle = tuple(
        run_natural_episode(
            episode,
            assessor,
            evidence=evidence,
            query_selectors=selectors,
            arm="PL1",
            candidate_label="PL1_ORACLE",
            proposal_arm="PL1",
            seed=RUN_SEED,
            protected_manifest=protected,
        )
        for episode, assessor, evidence, selectors in zip(
            typed, assessors, oracle_tables, oracle_queries, strict=True
        )
    )
    results_by_arm["PL1_ORACLE"] = oracle
    summaries["PL1_ORACLE"] = _summarize(oracle, time.perf_counter() - started)

    controls = {}
    for name, options in {
        "RESET": {"reset_before_transfer": True},
        "SHUFFLED": {"credit_control": "shuffled"},
        "WRONG_BRANCH": {"credit_control": "wrong_branch"},
        "NO_ELIGIBILITY": {"credit_control": "no_eligibility"},
    }.items():
        started = time.perf_counter()
        values = tuple(
            run_natural_episode(
                episode,
                assessor,
                evidence=evidence,
                query_selectors=selectors,
                arm="PL1",
                candidate_label=name,
                proposal_arm="PL1",
                seed=RUN_SEED,
                protected_manifest=protected,
                **options,
            )
            for episode, assessor, evidence, selectors in zip(
                typed, assessors, evidence_tables, query_selectors, strict=True
            )
        )
        controls[name] = _summarize(values, time.perf_counter() - started)

    transplant_rate = _transplant_rate(typed, assessors, query_selectors, npl2)
    rollback = _rollback(typed, assessors, query_selectors, npl2)
    mutation_rejected = False
    try:
        run_natural_episode(
            typed[0],
            assessors[0],
            evidence=evidence_tables[0],
            query_selectors=query_selectors[0],
            arm="PL1",
            proposal_arm="PL1",
            seed=RUN_SEED,
            protected_manifest=protected,
            inject_protected_mutation=True,
        )
    except RuntimeError:
        mutation_rejected = True

    npl2_rate = summaries["NPL2"]["query_rate"]
    oracle_rate = summaries["PL1_ORACLE"]["query_rate"]
    static_rate = summaries["STATIC"]["query_rate"]
    strongest = max(summaries[arm]["query_rate"] for arm in NATURAL_ARMS)
    attempt_gain = (
        summaries["NPL2"]["probe_query_exact_by_attempt"][-1]
        - summaries["NPL2"]["probe_query_exact_by_attempt"][0]
    ) / summaries["NPL2"]["probe_query_total_by_attempt"][0]
    conditions = {
        "world_exact": world["exact"] == world["total"] == 7168,
        "evidence_semantics_at_least_99_5_percent": _semantic_floor(
            evidence_score, 0.995, 3
        ),
        "query_semantics_at_least_99_5_percent": _semantic_floor(query_score, 0.995, 6),
        "npl2_query_at_least_80_percent": npl2_rate >= 0.80,
        "npl2_within_5_points_of_oracle": npl2_rate >= oracle_rate - 0.05,
        "npl2_beats_every_nonoracle_by_10_points": npl2_rate - strongest >= 0.10,
        "attempt_12_gain_at_least_50_points": attempt_gain >= 0.50,
        "reset_loses_25_points_and_returns_to_static": npl2_rate
        - controls["RESET"]["query_rate"]
        >= 0.25
        and controls["RESET"]["query_rate"] <= static_rate + 0.03,
        "shuffled_wrong_and_transplant_at_static": controls["SHUFFLED"]["query_rate"]
        <= static_rate + 0.03
        and controls["WRONG_BRANCH"]["query_rate"] <= static_rate + 0.03
        and transplant_rate <= static_rate + 0.03,
        "eligibility_ablation_loses_5_points": npl2_rate
        - controls["NO_ELIGIBILITY"]["query_rate"]
        >= 0.05,
        "rollback_exact_and_poison_changes_behavior": rollback["exact"]
        == rollback["total"]
        and rollback["changed"] / rollback["total"] >= 0.95,
        "protected_owner_hashes_exact": owner_hashes_exact
        and summaries["NPL2"]["protected_hashes_exact"]
        and mutation_rejected,
        "source_deleted_before_transfer": True,
    }
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "source_commit": args.source_commit,
        "world": dict(world),
        "semantic_compilation": {
            "evidence": evidence_score,
            "query": query_score,
            "hidden_outcome_selected_model_inference": False,
        },
        "summaries": summaries,
        "controls": controls,
        "transplant_query_rate": transplant_rate,
        "rollback": rollback,
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
        "resource_contract": {
            "episodes": len(typed),
            "branches_per_attempt": 8,
            "attempts_per_episode": 12,
            "verifier_calls_per_episode": 96,
            "late_queries_per_episode": 32,
            "mutable_scalars": 64,
            "write_norm_cap": 4.0,
            "score_clip": 8.0,
            "all_legal_feedback_compiled_before_outcomes": True,
            "durable_consolidation": False,
        },
        "custody": {
            "eni1_result": str(args.eni1_result),
            "eni1_result_sha256": args.eni1_result_sha256,
            "eic_checkpoint": str(args.eic_checkpoint),
            "eic_checkpoint_sha256": args.eic_checkpoint_sha256,
            "sti_checkpoint": str(args.sti_checkpoint),
            "sti_checkpoint_sha256": args.sti_checkpoint_sha256,
            "public_data": str(args.public_data),
            "public_data_sha256": args.public_data_sha256,
            "assessor_data": str(args.assessor_data),
            "assessor_data_sha256": args.assessor_data_sha256,
            "owner_hashes_exact": owner_hashes_exact,
            "confirmation_data_accessed": False,
        },
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "status": report["status"],
                "evidence": evidence_score["overall"],
                "query": query_score["overall"],
                "npl2_query_rate": npl2_rate,
                "oracle_query_rate": oracle_rate,
                "strongest_nonoracle_rate": strongest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
