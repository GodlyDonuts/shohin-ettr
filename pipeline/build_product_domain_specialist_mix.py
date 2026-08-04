#!/usr/bin/env python3
"""Build a verified rollout/replay mix for one routed task specialist."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "shohin-product-domain-specialist-mix-v1"
POSITIVE_SCHEMA = "shohin-product-rollout-positive-merge-v1"


class DomainSpecialistMixError(RuntimeError):
    """The domain-specialist corpus cannot satisfy its provenance contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _question(row: dict[str, Any]) -> str | None:
    for key in ("question", "problem", "prompt", "text", "input"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return None


def _response(row: dict[str, Any]) -> str | None:
    for key in ("response", "solution", "completion", "answer"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return None


def _identity(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    if not normalized:
        raise DomainSpecialistMixError("question identity is empty")
    return hashlib.sha256(normalized.encode()).hexdigest()


def _group(row: dict[str, Any]) -> str:
    return str(row.get("training_group") or row.get("domain") or "").strip().casefold()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise DomainSpecialistMixError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise DomainSpecialistMixError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_domain_mix(
    positive_report_path: Path,
    replay_path: Path,
    output: Path,
    report_output: Path,
    *,
    domain: str,
    replay_multiplier: float,
    seed: int,
) -> dict[str, Any]:
    domain = domain.strip().casefold()
    if not domain:
        raise DomainSpecialistMixError("domain is empty")
    if replay_multiplier <= 0:
        raise DomainSpecialistMixError("replay multiplier must be positive")

    positive_report = json.loads(positive_report_path.read_text())
    if (
        positive_report.get("schema") != POSITIVE_SCHEMA
        or positive_report.get("status") != "complete"
        or not positive_report.get("admitted")
    ):
        raise DomainSpecialistMixError("positive merge is not admitted")
    positives_path = Path(positive_report["positives_output"])
    if _sha256(positives_path) != positive_report.get("positives_sha256"):
        raise DomainSpecialistMixError("positive merge hash differs")

    counters: Counter[str] = Counter()
    positives: dict[str, dict[str, Any]] = {}
    with positives_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counters["positive_source_rows"] += 1
            row = json.loads(line)
            if _group(row) != domain:
                counters["positive_domain_drops"] += 1
                continue
            question = _question(row)
            if not question or not _response(row):
                raise DomainSpecialistMixError("positive row schema differs")
            identity = _identity(question)
            source_identity = str(row.get("source_identity_sha256") or "")
            if source_identity and source_identity != identity:
                raise DomainSpecialistMixError("positive source identity differs")
            if identity in positives:
                raise DomainSpecialistMixError("positive questions repeat")
            positives[identity] = row
    if not positives:
        raise DomainSpecialistMixError(f"no positive rows for domain {domain!r}")

    replay_candidates: dict[str, dict[str, Any]] = {}
    with replay_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counters["replay_source_rows"] += 1
            row = json.loads(line)
            if _group(row) != domain:
                counters["replay_domain_drops"] += 1
                continue
            question = _question(row)
            if not question or not _response(row):
                counters["replay_schema_drops"] += 1
                continue
            identity = _identity(question)
            if identity in positives:
                counters["replay_positive_overlap_drops"] += 1
                continue
            if identity in replay_candidates:
                counters["replay_duplicate_drops"] += 1
                continue
            replay_candidates[identity] = row

    requested_replay = int(round(len(positives) * replay_multiplier))
    ranked_replay = sorted(
        replay_candidates.items(),
        key=lambda item: hashlib.sha256(
            f"{seed}\0replay\0{item[0]}".encode()
        ).hexdigest(),
    )
    if len(ranked_replay) < requested_replay:
        raise DomainSpecialistMixError(
            f"domain replay capacity {len(ranked_replay)} is below requested {requested_replay}"
        )

    tagged = list(positives.items()) + ranked_replay[:requested_replay]
    tagged.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}\0output\0{item[0]}".encode()
        ).hexdigest()
    )
    if len({identity for identity, _ in tagged}) != len(tagged):
        raise DomainSpecialistMixError("output questions repeat")
    output_sha256 = _atomic_jsonl(output, [row for _, row in tagged])
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "domain": domain,
        "seed": seed,
        "replay_multiplier": replay_multiplier,
        "positive_report": str(positive_report_path.resolve()),
        "positive_report_sha256": _sha256(positive_report_path),
        "positives": str(positives_path.resolve()),
        "positives_sha256": positive_report["positives_sha256"],
        "positive_rows": len(positives),
        "replay": str(replay_path.resolve()),
        "replay_sha256": _sha256(replay_path),
        "replay_capacity": len(replay_candidates),
        "replay_rows_selected": requested_replay,
        "rows": len(tagged),
        "counters": dict(sorted(counters.items())),
        "output": str(output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(report_output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-report", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--replay-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_domain_mix(
        args.positive_report,
        args.replay,
        args.output,
        args.report_output,
        domain=args.domain,
        replay_multiplier=args.replay_multiplier,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
