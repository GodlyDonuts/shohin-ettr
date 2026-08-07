#!/usr/bin/env python3
"""Development-only semantic admission for conditional DIVERGE-NPL1."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_npl1_data import (
    DEVELOPMENT_COUNT,
    parse_program_surface,
    render_feedback,
    validate_natural_public_record,
)
from diverge_nve1_runtime import hard_role_permutation, tensorize_sources
from diverge_pl1_data import episode_from_assessor_record
from diverge_iem1_runtime import tensorize_queries
from eval_diverge_iem1 import sha256_path
from eval_diverge_sti1 import _load_sti1


SCHEMA = "shohin-diverge-npl1-semantic-development-v1"


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SystemExit(f"NPL1 development hash differs: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@torch.no_grad()
def _score_evidence(
    model,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    total = Counter()
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        ids, mask, bounds, symbols, numeric_targets, symbol_targets = tensorize_sources(
            batch, device
        )
        numeric_logits, symbol_logits = model.evidence_owner(ids, mask, bounds, symbols)
        for index, record in enumerate(batch):
            numeric = hard_role_permutation(numeric_logits[index]) == tuple(
                int(value) for value in numeric_targets[index].tolist()
            )
            symbolic = hard_role_permutation(symbol_logits[index]) == tuple(
                int(value) for value in symbol_targets[index].tolist()
            )
            renderer = str(int(record["renderer"]))
            for counter in (total, by_renderer[renderer]):
                counter["total"] += 1
                counter["numeric_exact"] += numeric
                counter["symbol_exact"] += symbolic
                counter["joint_exact"] += numeric and symbolic
    return {
        "overall": dict(total),
        "by_renderer": {key: dict(value) for key, value in sorted(by_renderer.items())},
    }


@torch.no_grad()
def _score_queries(
    model,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    total = Counter()
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        ids, mask, symbols, targets = tensorize_queries(batch, device)
        logits = model.forward_query(ids, mask, symbols)
        for index, record in enumerate(batch):
            exact = hard_role_permutation(logits[index]) == tuple(
                int(value) for value in targets[index].tolist()
            )
            renderer = str(int(record["renderer"]))
            for counter in (total, by_renderer[renderer]):
                counter["total"] += 1
                counter["exact"] += exact
    return {
        "overall": dict(total),
        "by_renderer": {key: dict(value) for key, value in sorted(by_renderer.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--public-data-sha256", required=True)
    parser.add_argument("--assessor-data", type=Path, required=True)
    parser.add_argument("--assessor-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NPL1 semantic output: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("NPL1 semantic admission requested unavailable CUDA")
    device = torch.device(args.device)

    public = _load_jsonl(args.public_data, args.public_data_sha256)
    assessor = _load_jsonl(args.assessor_data, args.assessor_data_sha256)
    if len(public) != DEVELOPMENT_COUNT or len(assessor) != DEVELOPMENT_COUNT:
        raise SystemExit("NPL1 semantic development count differs")
    model, checkpoint = _load_sti1(args.checkpoint, args.checkpoint_sha256, device)
    hashes_before = model.owner_hashes()

    world = Counter()
    evidence_records = []
    query_records = []
    for candidate, hidden in zip(public, assessor, strict=True):
        validate_natural_public_record(candidate)
        if hidden["public"] != candidate:
            raise SystemExit("NPL1 public/assessor surface differs")
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
        query_records.extend(
            {
                "source_text": query["source_text"],
                "symbols": symbol_table,
                "symbol_role_ids": query["symbol_role_ids"],
                "renderer": int(query["renderer"]),
            }
            for query in candidate["queries"]
        )

    evidence = _score_evidence(
        model, evidence_records, device=device, batch_size=args.batch_size
    )
    queries = _score_queries(
        model, query_records, device=device, batch_size=args.batch_size
    )
    evidence_overall = evidence["overall"]
    query_overall = queries["overall"]
    conditions = {
        "world_structural_exact": world["exact"] == world["total"] == 7168,
        "evidence_at_least_99_5_percent": evidence_overall["joint_exact"]
        / evidence_overall["total"]
        >= 0.995,
        "evidence_each_renderer_at_least_99_percent": all(
            values["joint_exact"] / values["total"] >= 0.99
            for values in evidence["by_renderer"].values()
        ),
        "query_at_least_99_5_percent": query_overall["exact"]
        / query_overall["total"]
        >= 0.995,
        "query_each_renderer_at_least_99_percent": all(
            values["exact"] / values["total"] >= 0.99
            for values in queries["by_renderer"].values()
        ),
        "protected_owner_hashes_exact": hashes_before == model.owner_hashes(),
    }
    result = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "source_commit": args.source_commit,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": args.checkpoint_sha256,
        "source_model_state_sha256": checkpoint["model_state_sha256"],
        "public_data": str(args.public_data),
        "public_data_sha256": args.public_data_sha256,
        "assessor_data": str(args.assessor_data),
        "assessor_data_sha256": args.assessor_data_sha256,
        "world": dict(world),
        "evidence": evidence,
        "query": queries,
        "owner_hashes_before": hashes_before,
        "owner_hashes_after": model.owner_hashes(),
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
        "confirmation_data_accessed": False,
        "plastic_updates": 0,
        "device": str(device),
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
                "query": result["query"]["overall"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
