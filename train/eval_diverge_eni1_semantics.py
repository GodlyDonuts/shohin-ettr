#!/usr/bin/env python3
"""Development-only admission for the EIC1 + NVE1 natural interface."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_cgl1_runtime import frozen_backbone_state_sha256
from diverge_eic1_runtime import render_claim_prompt
from diverge_nve1_data import symbol_occurrence_groups
from diverge_npl1_data import (
    DEVELOPMENT_COUNT,
    parse_program_surface,
    render_feedback,
    validate_natural_public_record,
)
from diverge_pl1_data import episode_from_assessor_record
from eval_diverge_cgl1 import _public, _score
from eval_diverge_eic1 import _load_model as _load_eic1
from eval_diverge_npl1_semantics import _load_jsonl, _score_evidence, _score_queries
from eval_diverge_pqi1 import sha256_path
from eval_diverge_sti1 import _load_sti1


SCHEMA = "shohin-diverge-eni1-semantic-development-v1"
WORLD_PROGRAMS = 7_168
EVIDENCE_TRANSACTIONS = 24_576
QUERY_TRANSACTIONS = 8_192


def _query_records(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    symbols = [str(value) for value in candidate["symbol_table"]]
    return [
        {
            "episode": str(candidate["episode_id"]),
            "stage": "QUERY",
            "mode": "register_query",
            "renderer": int(query["renderer"]),
            "source_text": str(query["source_text"]),
            "symbols": symbols,
            "symbol_role_ids": [int(value) for value in query["symbol_role_ids"]],
        }
        for query in candidate["queries"]
    ]


def _rename_query_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    renamed = []
    replacements = ("referentalpha", "referentbeta")
    for record in records:
        text = str(record["source_text"])
        symbols = [str(value) for value in record["symbols"]]
        groups = symbol_occurrence_groups(text, symbols)
        if len(groups) != 2 or any(value in symbols for value in replacements):
            raise SystemExit("ENI1 rename geometry differs")
        mapping = {groups[index][0]: replacements[index] for index in range(2)}
        placeholders = {
            symbol: f"__ENI1_ENTITY_{index}__"
            for index, symbol in enumerate(mapping)
        }
        for symbol, placeholder in placeholders.items():
            text = text.replace(symbol, placeholder)
        for symbol, placeholder in placeholders.items():
            text = text.replace(placeholder, mapping[symbol])
        item = dict(record)
        item["source_text"] = text
        item["symbols"] = [mapping.get(symbol, symbol) for symbol in symbols]
        renamed.append(item)
    return renamed


def _query_renderer_floor(score: Mapping[str, Any], ratio: float) -> bool:
    renderers = score.get("by_renderer", {})
    return len(renderers) == 6 and all(
        int(value.get("total", 0)) > 0
        and int(value.get("exact", 0)) / int(value["total"]) >= ratio
        for value in renderers.values()
    )


def gate_conditions(
    *,
    world_exact: int,
    evidence: Mapping[str, Any],
    normal: Mapping[str, Any],
    swapped: Mapping[str, Any],
    scrubbed: Mapping[str, Any],
    renamed: Mapping[str, Any],
    prompts_exact: bool,
    projection_error: float,
    protected_hashes_exact: bool,
) -> dict[str, bool]:
    evidence_overall = evidence["overall"]
    normal_exact = int(normal["overall"].get("exact", 0))
    normal_total = int(normal["overall"].get("total", 0))
    swapped_exact = int(swapped["overall"].get("exact", 0))
    swapped_total = int(swapped["overall"].get("total", 0))
    scrub_exact = int(scrubbed["overall"].get("exact", 0))
    return {
        "world_structural_exact": world_exact == WORLD_PROGRAMS,
        "evidence_at_least_99_5_percent": int(evidence_overall["total"])
        == EVIDENCE_TRANSACTIONS
        and int(evidence_overall["joint_exact"]) / EVIDENCE_TRANSACTIONS >= 0.995,
        "evidence_each_renderer_at_least_99_percent": len(evidence["by_renderer"])
        == 3
        and all(
            int(value["joint_exact"]) / int(value["total"]) >= 0.99
            for value in evidence["by_renderer"].values()
        ),
        "query_at_least_99_5_percent": normal_total == QUERY_TRANSACTIONS
        and normal_exact / normal_total >= 0.995,
        "query_each_renderer_at_least_99_percent": _query_renderer_floor(
            normal, 0.99
        ),
        "mapped_swap_at_least_99_5_percent": swapped_total == QUERY_TRANSACTIONS
        and swapped_exact / swapped_total >= 0.995,
        "mapped_swap_each_renderer_at_least_99_percent": _query_renderer_floor(
            swapped, 0.99
        ),
        "scrub_loses_at_least_40_points": normal_total == QUERY_TRANSACTIONS
        and (normal_exact - scrub_exact) / normal_total >= 0.40,
        "scrub_zero_margin": abs(float(scrubbed["mean_signed_margin"])) <= 1e-9,
        "entity_rename_prompts_exact": prompts_exact,
        "entity_rename_predictions_exact": normal["_predictions"]
        == renamed["_predictions"],
        "entity_rename_scores_bit_exact": torch.equal(
            normal["_scores"], renamed["_scores"]
        ),
        "projection_identity_exact": projection_error == 0.0,
        "protected_owner_hashes_exact": protected_hashes_exact,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing ENI1 result")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ENI1 requested unavailable CUDA")
    device = torch.device(args.device)

    public = _load_jsonl(args.public_data, args.public_data_sha256)
    assessor = _load_jsonl(args.assessor_data, args.assessor_data_sha256)
    if len(public) != DEVELOPMENT_COUNT or len(assessor) != DEVELOPMENT_COUNT:
        raise SystemExit("ENI1 development count differs")
    eic, eic_checkpoint = _load_eic1(
        args.eic_checkpoint,
        args.eic_checkpoint_sha256,
        args.base,
        args.base_sha256,
        args.tokenizer,
        args.tokenizer_sha256,
        device,
    )
    if eic_checkpoint["projection_mode"] != "involution":
        raise SystemExit("ENI1 requires the confirmed involution owner")
    sti, sti_checkpoint = _load_sti1(
        args.sti_checkpoint, args.sti_checkpoint_sha256, device
    )
    eic_frozen_before = frozen_backbone_state_sha256(eic.backbone)
    sti_hashes_before = sti.owner_hashes()

    world = Counter()
    evidence_records: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    for candidate, hidden in zip(public, assessor, strict=True):
        validate_natural_public_record(candidate)
        if hidden["public"] != candidate:
            raise SystemExit("ENI1 public/assessor surface differs")
        episode = episode_from_assessor_record(hidden["oracle"])
        aliases = tuple(str(value) for value in candidate["aliases"])
        registers_raw = tuple(str(value) for value in candidate["register_names"])
        registers = (registers_raw[0], registers_raw[1])
        surfaces = (*candidate["acquisition"], *candidate["transfer"])
        programs = (*episode.acquisition, *episode.transfer)
        for surface, program in zip(surfaces, programs, strict=True):
            initial, symbols = parse_program_surface(surface, aliases, registers)
            world["total"] += 1
            world["exact"] += initial == program.initial_state and symbols == program.symbols
        symbol_table = list(candidate["symbol_table"])
        for plan in candidate["feedback_plan"]:
            certificate_code = (3 * int(plan["attempt"]) + int(plan["branch"])) % 22
            evidence_records.append(
                {
                    "source_text": render_feedback(plan, certificate_code),
                    "symbols": symbol_table,
                    "numeric_role_ids": plan["numeric_role_ids"],
                    "symbol_role_ids": plan["symbol_role_ids"],
                    "renderer": int(plan["renderer"]),
                }
            )
        query_records.extend(_query_records(candidate))

    evidence = _score_evidence(
        sti, evidence_records, device=device, batch_size=max(128, args.batch_size)
    )
    rrg1_baseline = _score_queries(
        sti, query_records, device=device, batch_size=max(128, args.batch_size)
    )
    renamed_records = _rename_query_records(query_records)
    normal_prompts = [
        render_claim_prompt(row, candidate)
        for row in query_records
        for candidate in (0, 1)
    ]
    renamed_prompts = [
        render_claim_prompt(row, candidate)
        for row in renamed_records
        for candidate in (0, 1)
    ]
    normal = _score(eic, query_records, device=device, batch_size=args.batch_size)
    scrubbed = _score(
        eic,
        query_records,
        device=device,
        batch_size=args.batch_size,
        control="scrub_context",
    )
    swapped = _score(
        eic,
        query_records,
        device=device,
        batch_size=args.batch_size,
        control="swap_mentions",
        map_swapped_back=True,
    )
    renamed = _score(eic, renamed_records, device=device, batch_size=args.batch_size)
    projection_error = float(
        (normal["_scores"] - swapped["_scores"].flip(dims=(-1,))).abs().max()
    )
    protected_hashes_exact = (
        eic_frozen_before == frozen_backbone_state_sha256(eic.backbone)
        and sti_hashes_before == sti.owner_hashes()
    )
    conditions = gate_conditions(
        world_exact=int(world["exact"]),
        evidence=evidence,
        normal=normal,
        swapped=swapped,
        scrubbed=scrubbed,
        renamed=renamed,
        prompts_exact=normal_prompts == renamed_prompts,
        projection_error=projection_error,
        protected_hashes_exact=protected_hashes_exact,
    )
    result = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "world": dict(world),
        "evidence": evidence,
        "rrg1_query_baseline": rrg1_baseline,
        "eic1_query": {
            "normal": _public(normal),
            "scrub_context": _public(scrubbed),
            "mention_swap": _public(swapped),
            "entity_rename": {
                **_public(renamed),
                "prompts_bit_exact": normal_prompts == renamed_prompts,
                "prediction_mismatches": sum(
                    left != right
                    for left, right in zip(
                        normal["_predictions"], renamed["_predictions"], strict=True
                    )
                ),
                "max_absolute_score_difference": float(
                    (normal["_scores"] - renamed["_scores"]).abs().max()
                ),
            },
            "projection_identity_max_absolute_error": projection_error,
        },
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
        "eic_checkpoint": str(args.eic_checkpoint),
        "eic_checkpoint_sha256": args.eic_checkpoint_sha256,
        "eic_adapter_state_sha256": eic_checkpoint["adapter_state_sha256"],
        "eic_frozen_backbone_sha256": eic_frozen_before,
        "sti_checkpoint": str(args.sti_checkpoint),
        "sti_checkpoint_sha256": args.sti_checkpoint_sha256,
        "sti_model_state_sha256": sti_checkpoint["model_state_sha256"],
        "sti_owner_hashes": sti_hashes_before,
        "public_data": str(args.public_data),
        "public_data_sha256": args.public_data_sha256,
        "assessor_data": str(args.assessor_data),
        "assessor_data_sha256": args.assessor_data_sha256,
        "plastic_updates": 0,
        "confirmation_data_accessed": False,
    }
    _atomic_json(args.output, result)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "status": result["status"],
                "world": result["world"],
                "evidence": result["evidence"]["overall"],
                "rrg1_query": result["rrg1_query_baseline"]["overall"],
                "eic_query": result["eic1_query"]["normal"]["overall"],
                "eic_swap": result["eic1_query"]["mention_swap"]["overall"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
