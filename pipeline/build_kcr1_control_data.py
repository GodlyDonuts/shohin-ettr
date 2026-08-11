#!/usr/bin/env python3
"""Derive frozen KCR1 action-permuted and constant-restart controls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_kcr1_branch_data import render_prompt, sha256_file
from kcr1_branch_transducer import RESTART, execute_transaction, parse_transaction


SOURCE_SCHEMA = "shohin-kcr1-branch-train-v1"
SOURCE_REPORT_SCHEMA = "shohin-kcr1-branch-data-report-v1"
CONTROL_SCHEMA = "shohin-kcr1-control-train-v1"
REPORT_SCHEMA = "shohin-kcr1-control-data-report-v1"
PRESENTATIONS = ("verified_keep", "verified_continue", "natural_owner")


class KCR1ControlDataError(RuntimeError):
    """The KCR1 matched-control corpus differs."""


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise KCR1ControlDataError(f"refusing existing KCR1 control: {path}")
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
        raise KCR1ControlDataError(f"refusing existing KCR1 report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def draft_from_prompt(prompt: str) -> str:
    marker = "DRAFT:\n"
    prefix, separator, draft = prompt.rpartition(marker)
    if not separator or not prefix or not draft:
        raise KCR1ControlDataError("KCR1 prompt draft span differs")
    return draft


def load_source(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows or any(row.get("schema") != SOURCE_SCHEMA for row in rows):
        raise KCR1ControlDataError("KCR1 aligned source differs")
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise KCR1ControlDataError(f"refusing existing output root: {args.output}")
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    expected = source_report.get("output", {})
    if (
        source_report.get("schema") != SOURCE_REPORT_SCHEMA
        or source_report.get("status") != "complete"
        or source_report.get("zero_truncation") is not True
        or source_report.get("holdout_used") is not False
        or Path(expected.get("path", "")).resolve() != args.source.resolve()
        or expected.get("sha256") != sha256_file(args.source)
    ):
        raise KCR1ControlDataError("KCR1 aligned report differs")
    source_rows = load_source(args.source)
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in source_rows:
        source_identity = str(row.get("source_identity_sha256", ""))
        presentation = str(row.get("presentation", ""))
        if (
            len(source_identity) != 64
            or presentation not in PRESENTATIONS
            or presentation in grouped[source_identity]
        ):
            raise KCR1ControlDataError("KCR1 source group differs")
        grouped[source_identity][presentation] = row
    if any(set(group) != set(PRESENTATIONS) for group in grouped.values()):
        raise KCR1ControlDataError("KCR1 source group is incomplete")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    controls: dict[str, list[dict[str, Any]]] = {
        "action_permuted": [],
        "constant_restart": [],
    }
    counters: dict[str, Counter[str]] = {key: Counter() for key in controls}
    charged: Counter[str] = Counter()
    total_tokens: Counter[str] = Counter()
    maxima: dict[str, dict[str, int]] = {
        key: {"prompt": 0, "target": 0, "sequence": 0} for key in controls
    }
    # Dict insertion order preserves the exact aligned source/presentation order,
    # so the fixed reservoir seed sees the same source sequence in every arm.
    for source_identity in grouped:
        group = grouped[source_identity]
        continuation = group["verified_continue"]
        verified_target = execute_transaction(
            draft_from_prompt(continuation["question"]), continuation["response"]
        )
        target_sha = hashlib.sha256(verified_target.encode()).hexdigest()
        if target_sha != continuation.get("executed_target_sha256"):
            raise KCR1ControlDataError("KCR1 verified target hash differs")
        transactions = [group[presentation]["response"] for presentation in PRESENTATIONS]
        for index, presentation in enumerate(PRESENTATIONS):
            source = group[presentation]
            prompt = str(source["question"])
            draft = draft_from_prompt(prompt)
            values = {
                "action_permuted": transactions[(index + 1) % len(transactions)],
                "constant_restart": f"{RESTART}\n{verified_target}",
            }
            for control, response in values.items():
                action = parse_transaction(response).action
                executed = execute_transaction(draft, response)
                row = {
                    "schema": CONTROL_SCHEMA,
                    "identity_sha256": hashlib.sha256(
                        f"kcr1-control\0{control}\0{source_identity}\0{presentation}".encode()
                    ).hexdigest(),
                    "source_identity_sha256": source_identity,
                    "training_group": source["training_group"],
                    "presentation": presentation,
                    "control": control,
                    "question": prompt,
                    "response": response,
                    "action": action,
                    "executed_target_sha256": hashlib.sha256(executed.encode()).hexdigest(),
                    "runtime_fields": ["question"],
                    "assessor_fields_visible_to_model": False,
                }
                rendered = render_prompt(tokenizer, prompt)
                prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
                target_tokens = len(tokenizer.encode(response, add_special_tokens=False)) + 1
                sequence_tokens = prompt_tokens + target_tokens
                if sequence_tokens > args.max_sequence_length:
                    raise KCR1ControlDataError(f"KCR1 {control} row would truncate")
                controls[control].append(row)
                counters[control][f"action_{action}"] += 1
                charged[control] += target_tokens
                total_tokens[control] += sequence_tokens
                maxima[control]["prompt"] = max(maxima[control]["prompt"], prompt_tokens)
                maxima[control]["target"] = max(maxima[control]["target"], target_tokens)
                maxima[control]["sequence"] = max(
                    maxima[control]["sequence"], sequence_tokens
                )
    if any(len(rows) != len(source_rows) for rows in controls.values()):
        raise KCR1ControlDataError("KCR1 control row coverage differs")
    args.output.mkdir(parents=True)
    outputs: dict[str, dict[str, Any]] = {}
    for control, rows in controls.items():
        path = args.output / f"train_{control}.jsonl"
        outputs[control] = {
            "path": str(path.resolve()),
            "sha256": atomic_lines(path, rows),
            "rows": len(rows),
            "charged_target_tokens": charged[control],
            "total_sequence_tokens": total_tokens[control],
            "scan_counters": dict(counters[control]),
            "token_maxima": maxima[control],
        }
    result = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "source_report_sha256": sha256_file(args.source_report),
        "source_rows": len(source_rows),
        "source_groups": len(grouped),
        "max_sequence_length": args.max_sequence_length,
        "zero_truncation": True,
        "runtime_fields": ["question"],
        "assessor_fields_visible_to_model": False,
        "holdout_used": False,
        "outputs": outputs,
    }
    atomic_json(args.output / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    result = build(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
