#!/usr/bin/env python3
"""Build immutable counterfactually complete DIVERGE-RRG1 fit data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from diverge_ccr1_data import validate_ccr1_board_row
from diverge_iem1_data import validate_query_training_record
from diverge_nve1_data import validate_training_record as validate_evidence_source
from diverge_rrg1_data import (
    DATA_SEED,
    LEXICAL_FAMILIES,
    ROWS_PER_STAGE,
    derive_training_records,
    validate_counterfactual_completeness,
)
from diverge_srp1_data import validate_srp1_board_row


SCHEMA = "shohin-diverge-rrg1-data-report-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(
    path: Path,
    expected_sha256: str,
    validator: Callable[[Mapping[str, Any]], None],
    *,
    expected_rows: int,
) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"input hash differs: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validator(row)
            rows.append(row)
    if len(rows) != expected_rows:
        raise RuntimeError(f"input row count differs: {path}")
    return rows


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _board_sources(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    output = set()
    for row in rows:
        output.update(str(item["source_text"]) for item in row["natural_evidence"])
        output.update(
            str(item["source_text"]) for item in row["natural_queries"].values()
        )
    return output


def _stage_report(
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    family_order = Counter()
    family_form_order = Counter()
    family_original_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    source_by_identity = {
        str(row["identity_sha256"]): row for row in source_rows
    }
    pairs: defaultdict[str, set[int]] = defaultdict(set)
    for row in rows:
        family = int(row["family"])
        form = int(row["clause_form"])
        order = int(row["role_order"])
        family_order[f"{family}:{order}"] += 1
        family_form_order[f"{family}:{form}:{order}"] += 1
        pairs[str(row["pair_identity_sha256"])].add(order)
        original = source_by_identity[str(row["original_identity_sha256"])]
        family_original_renderer[str(family)][str(int(original["renderer"]))] += 1
    return {
        "rows": len(rows),
        "unique_rows": len({str(row["identity_sha256"]) for row in rows}),
        "unique_sources": len({str(row["source_text"]) for row in rows}),
        "pairs": len(pairs),
        "complete_pairs": sum(orders == {0, 1} for orders in pairs.values()),
        "family_order_counts": dict(sorted(family_order.items())),
        "family_form_order_counts": dict(sorted(family_form_order.items())),
        "family_original_renderer_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_original_renderer.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-source", type=Path, required=True)
    parser.add_argument("--evidence-source-sha256", required=True)
    parser.add_argument("--query-source", type=Path, required=True)
    parser.add_argument("--query-source-sha256", required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-sha256", required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--confirmation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DATA_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing RRG1 data output: {args.output}")
    if args.seed != DATA_SEED:
        raise SystemExit("RRG1 frozen data seed differs")

    evidence_source = _load_jsonl(
        args.evidence_source,
        args.evidence_source_sha256,
        validate_evidence_source,
        expected_rows=50_000,
    )
    query_source = _load_jsonl(
        args.query_source,
        args.query_source_sha256,
        validate_query_training_record,
        expected_rows=50_000,
    )
    development = _load_jsonl(
        args.development,
        args.development_sha256,
        validate_srp1_board_row,
        expected_rows=256,
    )
    confirmation = _load_jsonl(
        args.confirmation,
        args.confirmation_sha256,
        validate_ccr1_board_row,
        expected_rows=256,
    )
    evidence = derive_training_records(
        evidence_source, stage="EVIDENCE", seed=args.seed
    )
    query = derive_training_records(query_source, stage="QUERY", seed=args.seed)
    validate_counterfactual_completeness(evidence, stage="EVIDENCE")
    validate_counterfactual_completeness(query, stage="QUERY")

    generated_sources = {
        str(row["source_text"]) for row in (*evidence, *query)
    }
    development_sources = _board_sources(development)
    confirmation_sources = _board_sources(confirmation)
    training_entities = {
        str(symbol)
        for row in (*evidence_source, *query_source)
        for symbol in row["symbols"]
    }
    development_entities = {
        str(symbol) for row in development for symbol in row["tfs1"]["symbols"]
    }
    confirmation_entities = {
        str(symbol) for row in confirmation for symbol in row["tfs1"]["symbols"]
    }
    overlap = {
        "generated_with_development": len(generated_sources & development_sources),
        "generated_with_confirmation": len(generated_sources & confirmation_sources),
        "development_with_confirmation": len(
            development_sources & confirmation_sources
        ),
        "training_entities_with_development": len(
            training_entities & development_entities
        ),
        "training_entities_with_confirmation": len(
            training_entities & confirmation_entities
        ),
    }
    if any(overlap.values()):
        raise SystemExit(f"RRG1 split integrity failed: {overlap}")
    if len(evidence) != ROWS_PER_STAGE or len(query) != ROWS_PER_STAGE:
        raise SystemExit("RRG1 derived training geometry differs")

    args.output.mkdir(parents=True)
    evidence_path = args.output / "evidence_train.jsonl"
    query_path = args.output / "query_train.jsonl"
    _atomic_jsonl(evidence_path, evidence)
    _atomic_jsonl(query_path, query)
    report = {
        "schema": SCHEMA,
        "seed": DATA_SEED,
        "lexical_families": LEXICAL_FAMILIES,
        "model_score_used_for_selection": False,
        "pairing": "EVERY_SEMANTIC_ITEM_TARGET_FIRST_AND_DISTRACTOR_FIRST",
        "evidence": _stage_report(evidence, evidence_source),
        "query": _stage_report(query, query_source),
        "overlap": overlap,
        "evidence_source": str(args.evidence_source),
        "evidence_source_sha256": args.evidence_source_sha256,
        "query_source": str(args.query_source),
        "query_source_sha256": args.query_source_sha256,
        "development": str(args.development),
        "development_sha256": args.development_sha256,
        "confirmation": str(args.confirmation),
        "confirmation_sha256": args.confirmation_sha256,
        "evidence_training": str(evidence_path),
        "evidence_training_sha256": sha256_path(evidence_path),
        "query_training": str(query_path),
        "query_training_sha256": sha256_path(query_path),
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "evidence_training": str(evidence_path),
                "evidence_training_sha256": report["evidence_training_sha256"],
                "query_training": str(query_path),
                "query_training_sha256": report["query_training_sha256"],
                "report": str(report_path),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
