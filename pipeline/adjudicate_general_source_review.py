#!/usr/bin/env python3
"""Validate source-review labels and emit a text-free adjudication receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


DECISIONS = {"accept_core", "accept_residual", "reject"}
AUTHORITIES = {"human", "model_preliminary"}
HEX64 = re.compile(r"[0-9a-f]{64}")
SCORE_FIELDS = (
    "clarity",
    "correctness",
    "completeness",
    "educational_value",
    "originality",
    "real_user_utility",
)


class AdjudicationError(ValueError):
    """The review labels cannot support an admission decision."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AdjudicationError(f"{path}:{line_number}: malformed JSON") from exc
            if not isinstance(row, dict):
                raise AdjudicationError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def build_receipt(
    reviews: list[Mapping[str, Any]],
    labels: list[Mapping[str, Any]],
    *,
    review_sha256: str,
    labels_sha256: str,
    required_rows: int,
) -> dict[str, Any]:
    if required_rows < 100:
        raise AdjudicationError("production adjudication requires at least 100 rows")
    if len(reviews) != required_rows or len(labels) != required_rows:
        raise AdjudicationError(
            f"expected exactly {required_rows} reviews and labels, got "
            f"{len(reviews)} and {len(labels)}"
        )
    if not HEX64.fullmatch(review_sha256) or not HEX64.fullmatch(labels_sha256):
        raise AdjudicationError("review and label SHA-256 values must be lowercase hex")

    review_by_identity: dict[str, Mapping[str, Any]] = {}
    for row in reviews:
        identity = row.get("stable_identity_sha256")
        document = row.get("document_sha256")
        if not isinstance(identity, str) or not HEX64.fullmatch(identity):
            raise AdjudicationError("review has invalid stable identity")
        if not isinstance(document, str) or not HEX64.fullmatch(document):
            raise AdjudicationError(
                "semantic admission review must bind a nonempty text document"
            )
        if identity in review_by_identity:
            raise AdjudicationError("duplicate stable identity in review packet")
        review_by_identity[identity] = row

    decisions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    score_values: dict[str, list[float]] = defaultdict(list)
    authorities: Counter[str] = Counter()
    receipt_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in labels:
        identity = row.get("stable_identity_sha256")
        if identity not in review_by_identity:
            raise AdjudicationError("label identity is absent from review packet")
        if identity in seen:
            raise AdjudicationError("duplicate label identity")
        seen.add(str(identity))

        decision = row.get("decision")
        authority = row.get("adjudication_authority")
        if decision not in DECISIONS:
            raise AdjudicationError(f"invalid decision {decision!r}")
        if authority not in AUTHORITIES:
            raise AdjudicationError(f"invalid adjudication authority {authority!r}")
        decisions[str(decision)] += 1
        authorities[str(authority)] += 1

        row_scores: dict[str, int] = {}
        for field in SCORE_FIELDS:
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
                raise AdjudicationError(f"{field} must be an integer from 1 to 5")
            row_scores[field] = value
            score_values[field].append(float(value))

        row_reasons = row.get("reason_codes")
        if (
            not isinstance(row_reasons, list)
            or not row_reasons
            or not all(isinstance(reason, str) and reason for reason in row_reasons)
        ):
            raise AdjudicationError("reason_codes must be a nonempty string list")
        reasons.update(row_reasons)
        review = review_by_identity[str(identity)]
        receipt_rows.append(
            {
                "stable_identity_sha256": identity,
                "document_sha256": review.get("document_sha256"),
                "decision": decision,
                "adjudication_authority": authority,
                "scores": row_scores,
                "reason_codes": sorted(set(row_reasons)),
            }
        )

    if seen != set(review_by_identity):
        raise AdjudicationError("labels do not cover the complete review packet")

    complete_human_review = authorities == Counter({"human": required_rows})
    dataset_values = {str(row.get("dataset")) for row in reviews}
    config_values = {str(row.get("config")) for row in reviews}
    if len(dataset_values) != 1 or len(config_values) != 1:
        raise AdjudicationError("review packet mixes datasets or configurations")

    return {
        "schema": "shohin-general-source-adjudication-v1",
        "admission_status": (
            "human_semantic_gate_complete"
            if complete_human_review
            else "preliminary_not_training_admission"
        ),
        "training_admission_eligible": complete_human_review,
        "dataset": next(iter(dataset_values)),
        "config": next(iter(config_values)),
        "required_rows": required_rows,
        "review_packet_sha256": review_sha256,
        "labels_sha256": labels_sha256,
        "decision_counts": dict(decisions),
        "authority_counts": dict(authorities),
        "mean_scores": {
            field: _mean(score_values[field]) for field in SCORE_FIELDS
        },
        "reason_counts": dict(reasons.most_common()),
        "rows": sorted(receipt_rows, key=lambda row: row["stable_identity_sha256"]),
        "contains_document_text": False,
    }


def write_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise AdjudicationError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise AdjudicationError(f"refusing existing partial output {temporary}")
    with temporary.open("x") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--required-rows", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    review_path = Path(args.review)
    labels_path = Path(args.labels)
    receipt = build_receipt(
        load_jsonl(review_path),
        load_jsonl(labels_path),
        review_sha256=sha256_file(review_path),
        labels_sha256=sha256_file(labels_path),
        required_rows=args.required_rows,
    )
    write_no_replace(Path(args.out), receipt)
    print(
        json.dumps(
            {
                "dataset": receipt["dataset"],
                "config": receipt["config"],
                "status": receipt["admission_status"],
                "training_admission_eligible": receipt[
                    "training_admission_eligible"
                ],
                "out": args.out,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
