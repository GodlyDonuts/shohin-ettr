#!/usr/bin/env python3
"""Build matched MPR2 train and development curricula from trained-owner drafts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_mpr1_olmoe_revision_data import split_draft


SCHEMA = "shohin-mpr2-revision-train-v1"
REPORT_SCHEMA = "shohin-mpr2-revision-data-report-v1"
MPR1_REPORT_SCHEMA = "shohin-mpr1-revision-data-report-v1"
MTR_REPORT_SCHEMA = "shohin-idr1-revision-data-report-v1"
DRAFT_REPORT_SCHEMA = "shohin-mpr2-trained-owner-drafts-v1"
CANDIDATE_SCHEMA = "shohin-idr1-revision-candidate-v1"


class MPR2DataError(RuntimeError):
    """The MPR2 draft, target, split, or retention contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_lines(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise MPR2DataError(f"empty MPR2 input: {path}")
    return rows


def donor_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    result: dict[str, str] = {}
    for task, members in grouped.items():
        if len(members) < 2:
            raise MPR2DataError(f"singleton MPR2 task: {task}")
        ordered = sorted(members, key=lambda row: (row["draft_tokens"], row["source_id"]))
        for index, row in enumerate(ordered):
            candidates = ordered[max(0, index - 1) : index] + ordered[index + 1 : index + 2]
            donor = min(candidates, key=lambda candidate: (abs(candidate["draft_tokens"] - row["draft_tokens"]), candidate["source_id"]))
            result[row["source_id"]] = donor["source_id"]
    return result


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_bound(report: dict[str, Any], schema: str, path: Path, key: str) -> None:
    expected = report.get("outputs", {}).get(key, {})
    if report.get("schema") != schema or report.get("status") != "complete" or expected.get("sha256") != sha256_file(path) or Path(str(expected.get("path", ""))).resolve() != path.resolve():
        raise MPR2DataError(f"MPR2 bound input differs: {key}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise MPR2DataError(f"refusing existing output root: {args.output}")
    mpr1_report = json.loads(args.train_report.read_text())
    _validate_bound(mpr1_report, MPR1_REPORT_SCHEMA, args.train, "aligned")
    mtr_report = json.loads(args.development_report.read_text())
    _validate_bound(mtr_report, MTR_REPORT_SCHEMA, args.development, "development")
    draft_report = json.loads(args.train_draft_report.read_text())
    if (
        draft_report.get("schema") != DRAFT_REPORT_SCHEMA
        or draft_report.get("status") != "complete"
        or draft_report.get("output_sha256") != sha256_file(args.train_drafts)
        or Path(str(draft_report.get("output", ""))).resolve() != args.train_drafts.resolve()
    ):
        raise MPR2DataError("MPR2 trained-owner draft report differs")
    owner_report = json.loads(args.development_owner_report.read_text())
    if owner_report.get("schema") != "shohin-idr1-revision-evaluation-v1" or owner_report.get("status") != "complete" or owner_report.get("candidates_sha256") != sha256_file(args.development_owner_candidates):
        raise MPR2DataError("MPR2 development owner report differs")

    from transformers import AutoTokenizer
    from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    train_drafts = {row["source_identity_sha256"]: row for row in read_lines(args.train_drafts)}
    train_source = read_lines(args.train)
    parsed_train = []
    for row in train_source:
        source_id = str(row["source_identity_sha256"])
        draft_row = train_drafts.get(source_id)
        if draft_row is None:
            raise MPR2DataError("MPR2 training owner draft is absent")
        prefix, _, suffix = split_draft(str(row["question"]))
        parsed_train.append({"row": row, "source_id": source_id, "task": str(row["task"]), "prefix": prefix, "suffix": suffix, "draft": str(draft_row["completion"])})
    by_train: dict[str, dict[str, Any]] = {}
    for row in parsed_train:
        prior = by_train.setdefault(row["source_id"], row)
        if any(prior[key] != row[key] for key in ("task", "prefix", "suffix", "draft")):
            raise MPR2DataError("MPR2 repeated training source differs")
    for row in by_train.values():
        row["draft_tokens"] = len(tokenizer.encode(row["draft"], add_special_tokens=False))
    train_donors = donor_map(list(by_train.values()))

    owner_candidates = {row["identity_sha256"]: row for row in read_lines(args.development_owner_candidates)}
    development_source = read_lines(args.development)
    parsed_dev = []
    for row in development_source:
        identity = str(row["identity_sha256"])
        candidate = owner_candidates.get(identity)
        if candidate is None or candidate.get("schema") != CANDIDATE_SCHEMA:
            raise MPR2DataError("MPR2 development owner candidate is absent")
        prefix, _, suffix = split_draft(str(row["question"]))
        parsed_dev.append({"row": row, "source_id": identity, "task": str(row["task"]), "prefix": prefix, "suffix": suffix, "draft": str(candidate["completion"]), "owner": candidate})
    if len(owner_candidates) != len(parsed_dev):
        raise MPR2DataError("MPR2 development owner coverage differs")
    for row in parsed_dev:
        row["draft_tokens"] = len(tokenizer.encode(row["draft"], add_special_tokens=False))
    dev_donors = donor_map(parsed_dev)
    dev_by_id = {row["source_id"]: row for row in parsed_dev}

    def prompt_tokens(prompt: str) -> int:
        rendered = render_reasoning_messages(tokenizer, [{"role": "system", "content": PRODUCT_SYSTEM_PROMPT}, {"role": "user", "content": prompt}], enable_thinking=False)
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    train_aligned, train_shuffled = [], []
    rejected = 0
    charged_tokens = 0
    donor_deltas = []
    maximum = Counter()
    for item in parsed_train:
        donor = by_train[train_donors[item["source_id"]]]
        aligned_prompt = item["prefix"] + item["draft"] + item["suffix"]
        shuffled_prompt = item["prefix"] + donor["draft"] + item["suffix"]
        prompt_lengths = [prompt_tokens(aligned_prompt), prompt_tokens(shuffled_prompt)]
        target = str(item["row"]["response"])
        target_tokens = len(tokenizer.encode(target, add_special_tokens=False)) + 1
        if max(prompt_lengths) + target_tokens > 4096:
            rejected += 1
            continue
        common = {key: value for key, value in item["row"].items() if key not in {"identity_sha256", "question", "schema"}}
        common.update({"schema": SCHEMA, "task": item["task"], "complete_source_retained": True, "complete_draft_retained": True, "complete_target_retained": True})
        presentation = str(item["row"]["identity_sha256"])
        train_aligned.append({**common, "identity_sha256": hashlib.sha256(f"mpr2-aligned\0{presentation}".encode()).hexdigest(), "question": aligned_prompt, "draft_control": "aligned_trained_owner"})
        train_shuffled.append({**common, "identity_sha256": hashlib.sha256(f"mpr2-shuffled\0{presentation}".encode()).hexdigest(), "question": shuffled_prompt, "draft_control": "same_task_nearest_token_length", "draft_donor_identity_sha256": donor["source_id"]})
        charged_tokens += target_tokens
        maximum["prompt"] = max(maximum["prompt"], *prompt_lengths)
        maximum["target"] = max(maximum["target"], target_tokens)
        maximum["total"] = max(maximum["total"], max(prompt_lengths) + target_tokens)
        donor_deltas.append(abs(item["draft_tokens"] - donor["draft_tokens"]))
    if len(train_aligned) < int(0.95 * len(train_source)) or len(train_aligned) != len(train_shuffled):
        raise MPR2DataError("MPR2 matched training admission differs")

    dev_aligned, dev_shuffled = [], []
    for item in parsed_dev:
        donor = dev_by_id[dev_donors[item["source_id"]]]
        aligned_prompt = item["prefix"] + item["draft"] + item["suffix"]
        shuffled_prompt = item["prefix"] + donor["draft"] + item["suffix"]
        if max(prompt_tokens(aligned_prompt), prompt_tokens(shuffled_prompt)) + 768 > 4096:
            raise MPR2DataError("MPR2 complete development sequence exceeds 4096")
        base = dict(item["row"])
        base["internal_draft"] = dict(item["owner"])
        base["internal_draft"]["identity_sha256"] = item["source_id"]
        aligned = {**base, "question": aligned_prompt}
        shuffled = {**base, "question": shuffled_prompt, "internal_draft": {**donor["owner"], "identity_sha256": item["source_id"]}}
        dev_aligned.append(aligned)
        dev_shuffled.append(shuffled)

    args.output.mkdir(parents=True)
    outputs = {}
    for name, rows in (("train_aligned", train_aligned), ("train_shuffled", train_shuffled), ("development_aligned", dev_aligned), ("development_shuffled", dev_shuffled)):
        path = args.output / f"{name}.jsonl"
        outputs[name] = {"path": str(path.resolve()), "sha256": atomic_lines(path, rows), "rows": len(rows)}
    for arm in ("aligned", "shuffled"):
        source = outputs[f"development_{arm}"]
        atomic_json(args.output / f"development_{arm}_report.json", {"schema": MTR_REPORT_SCHEMA, "status": "complete", "internal_draft_visible": True, "external_candidate_text_visible": False, "runtime_fields": ["question"], "outputs": {"development": source}, "holdout_used": False})
    deltas = sorted(donor_deltas)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_root": str(args.model_root.resolve()),
        "owner_checkpoint_sha256": draft_report["owner_checkpoint_sha256"],
        "owner_development_score": int(owner_report["metrics"]["overall"]["generated_correct"]),
        "owner_development_domains": {task: int(owner_report["metrics"][task]["generated_correct"]) for task in ("math500", "bbh_logic", "mbpp")},
        "train_source_sha256": sha256_file(args.train),
        "train_draft_report_sha256": sha256_file(args.train_draft_report),
        "development_source_sha256": sha256_file(args.development),
        "development_owner_report_sha256": sha256_file(args.development_owner_report),
        "holdout_used": False,
        "max_sequence_length": 4096,
        "complete_retention": True,
        "target_multiset_exactly_matched": [row["response"] for row in train_aligned] == [row["response"] for row in train_shuffled],
        "zero_source_donor_identity_matches": all(a["source_identity_sha256"] != s["draft_donor_identity_sha256"] for a, s in zip(train_aligned, train_shuffled, strict=True)),
        "same_task_donors": True,
        "trained_olmoe_owned_drafts": True,
        "train_source_rows": len(train_source),
        "unique_train_sources": len(by_train),
        "admitted_train_rows_per_arm": len(train_aligned),
        "rejected_train_rows": rejected,
        "development_rows_per_arm": len(dev_aligned),
        "charged_target_tokens_per_arm": charged_tokens,
        "maximum_tokens": dict(maximum),
        "donor_token_delta_p95": deltas[min(len(deltas) - 1, int(0.95 * len(deltas)))],
        "task_counts_per_arm": dict(Counter(row["task"] for row in train_aligned)),
        "outputs": outputs,
    }
    if not report["target_multiset_exactly_matched"] or not report["zero_source_donor_identity_matches"]:
        raise MPR2DataError("MPR2 causal matching differs")
    atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--train-drafts", type=Path, required=True)
    parser.add_argument("--train-draft-report", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--development-owner-candidates", type=Path, required=True)
    parser.add_argument("--development-owner-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

