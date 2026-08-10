#!/usr/bin/env python3
"""Materialize verified DSET transactions from immutable on-policy proposals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

from dset1_edit_transducer import KEEP, REPLACE_LAST, execute_script, parse_script, render_script
from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages
from train_dset1_span_edit import DATA_REPORT_SCHEMA, DATA_SCHEMA, sha256_file


FRET_SCHEMA = "shohin-fret1-always-rewrite-evaluation-v1"


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_proposals(paths: list[Path]) -> tuple[dict[str, dict], list[dict]]:
    if len(paths) != 16:
        raise RuntimeError("OCET1 proposal shard count differs")
    proposals, receipts, shards = {}, [], set()
    for path in paths:
        report = json.loads(path.read_text())
        if (
            report.get("schema") != FRET_SCHEMA
            or report.get("status") != "complete"
            or report.get("holdout_used") is not False
            or report.get("arm") != "aligned"
            or int(report.get("shard_count", -1)) != 16
        ):
            raise RuntimeError("OCET1 proposal report differs")
        shard = int(report["shard_index"])
        if shard in shards:
            raise RuntimeError("OCET1 duplicate proposal shard")
        shards.add(shard)
        receipts.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        for row in report["results"]:
            identity = str(row["identity_sha256"])
            if identity in proposals:
                raise RuntimeError("OCET1 duplicate proposal identity")
            proposals[identity] = row
    if shards != set(range(16)) or len(proposals) != 15278:
        raise RuntimeError("OCET1 proposal coverage differs")
    return proposals, receipts


def derive_row(row: dict, proposal: dict) -> tuple[dict, str]:
    original_draft = str(row["draft"])
    final = str(row["final_response"])
    candidate = str(proposal.get("executed_trajectory") or "")
    mode = "on_policy"
    if not candidate:
        candidate = original_draft
        script = str(row["script"])
        mode = "fallback_original"
    elif candidate == final:
        script = render_script(KEEP)
    else:
        old = str(proposal.get("new_surface") or "")
        if not old or candidate.rfind(old) < 0:
            candidate = original_draft
            script = str(row["script"])
            mode = "fallback_original"
        else:
            script = render_script(REPLACE_LAST, old, str(row["gold_answer"]))
            if execute_script(candidate, parse_script(script)) != final:
                candidate = original_draft
                script = str(row["script"])
                mode = "fallback_original"
    question = str(row["question"])
    if question.count(original_draft) != 1:
        raise RuntimeError("OCET1 source/draft boundary differs")
    question = question.replace(original_draft, candidate, 1)
    parsed = parse_script(script)
    if execute_script(candidate, parsed) != final:
        raise RuntimeError("OCET1 derived transaction differs")
    output = dict(row)
    output.update(
        {
            "identity_sha256": digest_text(f"ocet1\0{row['identity_sha256']}\0{digest_text(candidate)}"),
            "question": question,
            "draft": candidate,
            "draft_sha256": digest_text(candidate),
            "script": script,
            "action": parsed.action,
            "old_surface": parsed.old,
            "new_surface": parsed.new,
            "on_policy_mode": mode,
            "proposal_sha256": digest_text(str(proposal.get("completion", ""))),
            "proposal_execution_correct": candidate == final,
        }
    )
    return output, mode


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise RuntimeError("OCET1 output exists")
    source_report = json.loads(args.source_report.read_text())
    expected = source_report.get("outputs", {}).get("train", {})
    if (
        source_report.get("schema") != DATA_REPORT_SCHEMA
        or source_report.get("status") != "complete"
        or source_report.get("holdout_used") is not False
        or Path(str(expected.get("path", ""))).resolve() != args.source.resolve()
        or expected.get("sha256") != sha256_file(args.source)
        or int(expected.get("rows", -1)) != 15278
    ):
        raise RuntimeError("OCET1 source report differs")
    original = [json.loads(line) for line in args.source.read_text().splitlines() if line]
    if any(row.get("schema") != DATA_SCHEMA for row in original):
        raise RuntimeError("OCET1 source row differs")
    proposals, proposal_receipts = load_proposals(args.proposals)
    if {str(row["identity_sha256"]) for row in original} != set(proposals):
        raise RuntimeError("OCET1 source/proposal identities differ")
    rows, modes = [], Counter()
    grouped = defaultdict(list)
    for row in original:
        converted, mode = derive_row(row, proposals[str(row["identity_sha256"])])
        grouped[str(converted["pair_identity_sha256"])].append(converted)
        modes[mode] += 1
    for pair_id in sorted(grouped):
        pair = sorted(grouped[pair_id], key=lambda item: item["pair_member"])
        if len(pair) != 2 or {item["pair_member"] for item in pair} != {"clean", "fault"}:
            raise RuntimeError("OCET1 pair geometry differs")
        pair[0]["swapped_script"], pair[1]["swapped_script"] = pair[1]["script"], pair[0]["script"]
        rows.extend(pair)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    maxima = Counter()
    action_counts = Counter()
    for row in rows:
        rendered = render_reasoning_messages(
            tokenizer,
            [{"role": "system", "content": PRODUCT_SYSTEM_PROMPT}, {"role": "user", "content": row["question"]}],
            enable_thinking=False,
        )
        prompt = tokenizer.encode(rendered, add_special_tokens=False)
        script = tokenizer.encode(row["script"], add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(script) > 32 or len(prompt) + len(script) > 4096:
            raise RuntimeError("OCET1 complete retention differs")
        maxima["prompt"] = max(maxima["prompt"], len(prompt))
        maxima["script"] = max(maxima["script"], len(script))
        maxima["complete"] = max(maxima["complete"], len(prompt) + len(script))
        action_counts[row["action"]] += 1
    args.output.mkdir(parents=True)
    data = args.output / "train.jsonl"
    atomic_jsonl(data, rows)
    report = {
        "schema": DATA_REPORT_SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "complete_retention": True,
        "train_diagnostic_source_overlap": 0,
        "max_script_tokens": 32,
        "max_sequence_length": 4096,
        "source_report_sha256": sha256_file(args.source_report),
        "source_train_sha256": sha256_file(args.source),
        "model_config_sha256": sha256_file(args.model_root / "config.json"),
        "proposal_receipts": proposal_receipts,
        "modes": dict(modes),
        "actions": dict(action_counts),
        "maximum_tokens": dict(maxima),
        "outputs": {"train": {"path": str(data.resolve()), "sha256": sha256_file(data), "rows": len(rows), "sources": len(grouped)}},
    }
    atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
