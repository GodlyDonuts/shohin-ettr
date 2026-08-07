#!/usr/bin/env python3
"""Read-only class-balance attribution for the closed QTE1 scrub gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from diverge_gti1_runtime import canonical_query_text, expected_transaction
from eval_diverge_ccr1 import _referent_records


SCHEMA = "shohin-diverge-qte1-scrub-balance-attribution-v1"


class QTE1ScrubAttributionError(RuntimeError):
    """The immutable QTE1 board or report cannot support exact attribution."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--board-sha256", required=True)
    parser.add_argument("--qte1-result", type=Path, required=True)
    parser.add_argument("--qte1-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing QTE1 attribution: {args.output}")
    for path, expected, label in (
        (args.board, args.board_sha256, "board"),
        (args.qte1_result, args.qte1_result_sha256, "result"),
    ):
        if sha256_path(path) != expected:
            raise QTE1ScrubAttributionError(f"QTE1 {label} hash differs")

    with args.board.open(encoding="utf-8") as handle:
        board = [json.loads(line) for line in handle]
    records = [row for row in _referent_records(board) if row["stage"] == "QUERY"]
    if len(records) != 768:
        raise QTE1ScrubAttributionError("QTE1 query count differs")
    result = json.loads(args.qte1_result.read_text(encoding="utf-8"))
    scrub = result["controls"]["scrub_context"]

    expected = Counter(expected_transaction(row) for row in records)
    by_renderer: defaultdict[int, list[int]] = defaultdict(list)
    scrub_texts = Counter()
    for row in records:
        by_renderer[int(row["renderer"])].append(expected_transaction(row))
        scrub_texts[canonical_query_text(row, control="scrub_context")] += 1
    if len(scrub_texts) != 1:
        raise QTE1ScrubAttributionError("QTE1 scrubbed prompts are not identical")

    inferred_predictions = Counter()
    inference_receipts = {}
    for renderer, labels in sorted(by_renderer.items()):
        label_counts = Counter(labels)
        if len(label_counts) != 1:
            raise QTE1ScrubAttributionError(
                "QTE1 renderer is not label-constant; aggregate inference is invalid"
            )
        label = next(iter(label_counts))
        reported = scrub["by_renderer"][str(renderer)]
        total = int(reported["total"])
        exact = int(reported["exact"])
        if total != len(labels) or exact not in (0, total):
            raise QTE1ScrubAttributionError(
                "QTE1 aggregate does not identify a unique renderer prediction"
            )
        prediction = label if exact == total else 1 - label
        inferred_predictions[prediction] += total
        inference_receipts[str(renderer)] = {
            "expected_transaction": label,
            "inferred_prediction": prediction,
            "exact": exact,
            "total": total,
        }

    majority = max(expected.values())
    reported_exact = int(scrub["overall"]["exact"])
    if reported_exact != majority or inferred_predictions != Counter({0: 768}):
        raise QTE1ScrubAttributionError(
            "QTE1 scrub result is not the inferred uniform-alpha majority baseline"
        )
    report = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Read-only accounting attribution of the already-closed QTE1 scrub "
            "condition; it does not alter or promote QTE1."
        ),
        "board": str(args.board),
        "board_sha256": args.board_sha256,
        "qte1_result": str(args.qte1_result),
        "qte1_result_sha256": args.qte1_result_sha256,
        "query_rows": len(records),
        "scrub_text": next(iter(scrub_texts)),
        "scrub_text_count": next(iter(scrub_texts.values())),
        "expected_transaction_counts": {
            str(key): value for key, value in sorted(expected.items())
        },
        "inferred_prediction_counts": {
            str(key): value for key, value in sorted(inferred_predictions.items())
        },
        "renderer_receipts": inference_receipts,
        "reported_scrub_exact": reported_exact,
        "majority_class_exact": majority,
        "attribution": (
            "All source-scrubbed prompts are identical. QTE1 predicts transaction "
            "0 (alpha) for all 768 rows; the sealed board contains 512 alpha and "
            "256 beta targets, so 512/768 equals the majority-class baseline."
        ),
        "decision": (
            "retain_qte1_as_a_capability_floor_observation_but_keep_the_frozen_"
            "confirmation_formally_failed;_do_not_retry_or_promote"
        ),
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "reported_scrub_exact": reported_exact,
                "majority_class_exact": majority,
                "inferred_prediction_counts": dict(inferred_predictions),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
