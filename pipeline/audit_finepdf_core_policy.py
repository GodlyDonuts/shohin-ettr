#!/usr/bin/env python3
"""Apply the FinePDF candidate policy to a private review packet.

The resulting report contains identities and aggregate statistics but no
document text. It is evidence for filter design, never a semantic or training
admission.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipeline.finepdf_core_policy import (
    POLICY_SCHEMA,
    classify_finepdf_candidate,
)


REPORT_SCHEMA = "shohin-finepdf-candidate-policy-analysis-v1"
PACKET_SCHEMA = "shohin-private-selected-source-review-v1"


class FinePdfPolicyAuditError(ValueError):
    """The private packet or policy analysis contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material.pop("payload_sha256", None)
    return hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def select(fraction: float) -> float:
        index = round(fraction * (len(ordered) - 1))
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "p10": select(0.10),
        "p50": select(0.50),
        "p90": select(0.90),
        "max": float(ordered[-1]),
    }


def _review_priority(identity: str, tier: str) -> str:
    return hashlib.sha256(
        f"shohin-finepdf-policy-review-v1\x1f{tier}\x1f{identity}".encode("ascii")
    ).hexdigest()


def audit_packet(
    packet_path: Path,
    *,
    review_rows_per_tier: int,
    policy_path: Path,
) -> dict[str, Any]:
    if review_rows_per_tier < 1:
        raise FinePdfPolicyAuditError("review rows per tier must be positive")
    seen: set[str] = set()
    tier_counts: Counter[str] = Counter()
    tier_tokens: Counter[str] = Counter()
    tier_words: Counter[str] = Counter()
    tier_domains: dict[str, set[str]] = {}
    reason_counts: Counter[str] = Counter()
    score_values: dict[str, list[float]] = {}
    review_candidates: dict[str, list[tuple[str, str]]] = {}
    dataset = config = None

    with packet_path.open(encoding="ascii") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FinePdfPolicyAuditError(
                    f"packet row {line_number} is malformed"
                ) from exc
            if not isinstance(row, dict) or row.get("schema") != PACKET_SCHEMA:
                raise FinePdfPolicyAuditError("private packet schema differs")
            identity = row.get("stable_identity_sha256")
            if (
                not isinstance(identity, str)
                or len(identity) != 64
                or identity in seen
            ):
                raise FinePdfPolicyAuditError("packet identity differs")
            seen.add(identity)
            dataset = dataset or row.get("dataset")
            config = config or row.get("config")
            if dataset != row.get("dataset") or config != row.get("config"):
                raise FinePdfPolicyAuditError("packet source identity drifts")
            text = row.get("review_text")
            metadata = row.get("metadata")
            selection = row.get("selection")
            if (
                not isinstance(text, str)
                or not isinstance(metadata, dict)
                or not isinstance(selection, dict)
            ):
                raise FinePdfPolicyAuditError("packet row fields differ")
            domain = selection.get("domain")
            decision = classify_finepdf_candidate(
                text=text,
                metadata=metadata,
                domain=domain if isinstance(domain, str) else None,
            )
            if decision.schema != POLICY_SCHEMA:
                raise FinePdfPolicyAuditError("policy schema differs")
            tier = decision.tier
            tokens = selection.get("tokens")
            if not isinstance(tokens, int) or tokens < 1:
                raise FinePdfPolicyAuditError("packet token count differs")
            tier_counts[tier] += 1
            tier_tokens[tier] += tokens
            tier_words[tier] += decision.word_count
            tier_domains.setdefault(tier, set()).add(str(domain))
            reason_counts.update(decision.reason_codes)
            score_values.setdefault(tier, []).append(decision.education_score_mean)
            review_candidates.setdefault(tier, []).append(
                (_review_priority(identity, tier), identity)
            )

    if not seen:
        raise FinePdfPolicyAuditError("private packet is empty")
    tiers = sorted(tier_counts)
    review_selection = {
        tier: [
            identity
            for _priority, identity in sorted(review_candidates[tier])[
                :review_rows_per_tier
            ]
        ]
        for tier in tiers
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "analysis_only_not_training_admission",
        "contains_document_text": False,
        "dataset": dataset,
        "config": config,
        "packet": {
            "bytes": packet_path.stat().st_size,
            "rows": len(seen),
            "sha256": sha256_file(packet_path),
        },
        "policy": {
            "schema": POLICY_SCHEMA,
            "path": policy_path.name,
            "sha256": sha256_file(policy_path),
        },
        "tiers": {
            tier: {
                "documents": tier_counts[tier],
                "tokens": tier_tokens[tier],
                "words_in_bounded_excerpt": tier_words[tier],
                "unique_domains": len(tier_domains[tier]),
                "education_score_mean_quantiles": _quantiles(score_values[tier]),
            }
            for tier in tiers
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "review_selection": review_selection,
        "review_rows_per_tier": review_rows_per_tier,
    }
    report["payload_sha256"] = canonical_payload_sha256(report)
    return report


def write_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FinePdfPolicyAuditError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="ascii") as output:
        json.dump(payload, output, indent=2, sort_keys=True, ensure_ascii=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-packet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review-rows-per-tier", type=int, default=100)
    args = parser.parse_args()
    policy_path = Path(__file__).with_name("finepdf_core_policy.py").resolve()
    report = audit_packet(
        args.private_packet.resolve(),
        review_rows_per_tier=args.review_rows_per_tier,
        policy_path=policy_path,
    )
    write_no_replace(args.out.resolve(), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
