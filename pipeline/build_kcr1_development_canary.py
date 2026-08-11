#!/usr/bin/env python3
"""Build the source-disjoint three-state KCR1 transaction canary."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_kcr1_branch_data import continuation_split, render_prompt, sha256_file
from build_vcr1_revision_data import source_task_prompt
from kcr1_branch_transducer import CONTINUE, KEEP, RESTART, execute_transaction, kcr1_prompt


SCHEMA = "shohin-kcr1-development-canary-v1"
REPORT_SCHEMA = "shohin-kcr1-development-canary-report-v1"
IDR_SCHEMA = "shohin-idr1-revision-eval-v1"
IDR_REPORT_SCHEMA = "shohin-idr1-revision-data-report-v1"


class KCR1CanaryError(RuntimeError):
    """The source-disjoint KCR1 canary contract differs."""


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise KCR1CanaryError(f"refusing existing KCR1 canary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise KCR1CanaryError(f"refusing existing KCR1 canary report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_development(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != IDR_SCHEMA
                or row.get("split") != "development"
                or not isinstance(identity, str)
                or identity in identities
                or row.get("runtime_fields") != ["question"]
            ):
                raise KCR1CanaryError("IDR1 development row differs")
            identities.add(identity)
            draft = row.get("internal_draft")
            candidates = row.get("candidates")
            if (
                not isinstance(draft, dict)
                or draft.get("identity_sha256") != identity
                or not isinstance(candidates, list)
                or len(candidates) != 2
                or not isinstance(row.get("assessor"), dict)
            ):
                raise KCR1CanaryError("IDR1 canary source binding differs")
            rows.append(row)
    if len(rows) != 1289:
        raise KCR1CanaryError("IDR1 development population differs")
    return rows


def verified_trajectory(row: dict[str, Any]) -> str | None:
    candidates = [
        candidate
        for candidate in row["candidates"]
        if candidate.get("correct") is True
        and isinstance(candidate.get("completion"), str)
        and candidate["completion"]
    ]
    if not candidates:
        return None
    chosen = min(
        candidates,
        key=lambda value: (len(value["completion"]), str(value.get("lineage", ""))),
    )
    return str(chosen["completion"])


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise KCR1CanaryError(f"refusing existing output root: {args.output}")
    report = json.loads(args.development_report.read_text(encoding="utf-8"))
    expected = report.get("outputs", {}).get("development", {})
    if (
        report.get("schema") != IDR_REPORT_SCHEMA
        or report.get("status") != "complete"
        or Path(expected.get("path", "")).resolve() != args.development.resolve()
        or expected.get("sha256") != sha256_file(args.development)
        or expected.get("rows") != 1289
    ):
        raise KCR1CanaryError("IDR1 development report differs")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    prompt_token_max = target_token_max = 0
    for source_row in load_development(args.development):
        identity = str(source_row["identity_sha256"])
        verified = verified_trajectory(source_row)
        if verified is None:
            counters["no_verified_trajectory"] += 1
            continue
        try:
            prefix, suffix = continuation_split(verified)
        except Exception:
            counters["unsplittable_verified_trajectory"] += 1
            continue
        natural = source_row["internal_draft"]
        natural_draft = str(natural.get("completion", ""))
        exhausted = natural.get("max_token_exhausted") is True
        natural_keep = (
            natural.get("correct") is True
            and not exhausted
            and source_row.get("task") != "mbpp"
        )
        source_prompt = source_task_prompt(source_row["assessor"])
        presentations = (
            ("verified_keep", verified, False, KEEP, "", verified),
            ("verified_continue", prefix, True, CONTINUE, suffix, verified),
            (
                "natural_owner",
                natural_draft,
                exhausted,
                KEEP if natural_keep else RESTART,
                "" if natural_keep else verified,
                natural_draft if natural_keep else verified,
            ),
        )
        staged: list[tuple[dict[str, Any], int, int]] = []
        for presentation, draft, cutoff, action, payload, expected_execution in presentations:
            transaction = action if action == KEEP else f"{action}\n{payload}"
            if execute_transaction(draft, transaction) != expected_execution:
                raise KCR1CanaryError("KCR1 canary transaction does not round-trip")
            prompt = kcr1_prompt(
                source_prompt,
                draft,
                exhausted=cutoff,
                task="code" if source_row["task"] == "mbpp" else "reasoning",
            )
            rendered = render_prompt(tokenizer, prompt)
            prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
            target_tokens = len(tokenizer.encode(transaction, add_special_tokens=False)) + 1
            if prompt_tokens + target_tokens > args.max_sequence_length:
                counters[f"overflow_{presentation}"] += 1
                staged = []
                break
            staged.append(
                (
                    {
                        "schema": SCHEMA,
                        "identity_sha256": hashlib.sha256(
                            f"kcr1-canary\0{identity}\0{presentation}".encode()
                        ).hexdigest(),
                        "source_identity_sha256": identity,
                        "split": "development",
                        "task": source_row["task"],
                        "presentation": presentation,
                        "question": prompt,
                        "draft": draft,
                        "expected_action": action,
                        "expected_transaction_sha256": hashlib.sha256(
                            transaction.encode()
                        ).hexdigest(),
                        "expected_execution": expected_execution,
                        "assessor": source_row["assessor"],
                        "runtime_fields": ["question"],
                        "assessor_fields_visible_to_model": False,
                    },
                    prompt_tokens,
                    target_tokens,
                )
            )
        if len(staged) != 3:
            continue
        for row, prompt_tokens, target_tokens in staged:
            rows.append(row)
            counters[f"action_{row['expected_action']}"] += 1
            counters[f"presentation_{row['presentation']}"] += 1
            prompt_token_max = max(prompt_token_max, prompt_tokens)
            target_token_max = max(target_token_max, target_tokens)

    source_count = len({row["source_identity_sha256"] for row in rows})
    if source_count < 400 or len(rows) != 3 * source_count:
        raise KCR1CanaryError("KCR1 canary source/branch coverage is insufficient")
    if len({row["identity_sha256"] for row in rows}) != len(rows):
        raise KCR1CanaryError("KCR1 canary identity is duplicated")
    args.output.mkdir(parents=True)
    data_path = args.output / "development.jsonl"
    data_sha256 = atomic_lines(data_path, rows)
    result = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "split": "development",
        "development": str(args.development.resolve()),
        "development_sha256": sha256_file(args.development),
        "development_report_sha256": sha256_file(args.development_report),
        "source_rows": 1289,
        "admitted_sources": source_count,
        "presentations": len(rows),
        "presentations_per_source": 3,
        "scan_counters": dict(counters),
        "prompt_token_max": prompt_token_max,
        "target_token_max": target_token_max,
        "max_sequence_length": args.max_sequence_length,
        "zero_truncation": True,
        "runtime_fields": ["question"],
        "assessor_fields_visible_to_model": False,
        "holdout_used": False,
        "output": {
            "path": str(data_path.resolve()),
            "sha256": data_sha256,
            "rows": len(rows),
        },
    }
    atomic_json(args.output / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    result = build(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
