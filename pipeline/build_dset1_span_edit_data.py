#!/usr/bin/env python3
"""Convert immutable DSEO1 pairs into verified DSET1 edit-script pairs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_dseo1_paired_data import (
    ANSWER_SENTENCE,
    DSEO1_RESPONSE_INSTRUCTION,
    OBR1_REPORT_SCHEMA,
    OBR1_SCHEMA,
    REPORT_SCHEMA as DSEO1_REPORT_SCHEMA,
    SCHEMA as DSEO1_SCHEMA,
    SOURCE_REPORT_SCHEMA,
    _bound_output,
    boxed_inner_span,
    make_fault,
    presentations,
    read_jsonl,
    split_members,
)
from dset1_edit_transducer import KEEP, REPLACE_LAST, render_script
from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages


SCHEMA = "shohin-dset1-span-edit-presentation-v1"
REPORT_SCHEMA = "shohin-dset1-span-edit-data-report-v1"
INSTRUCTION = (
    "\n\nEmit exactly one edit script. Use one line `<KEEP>` when the draft is "
    "already correct. Otherwise emit exactly three nonempty lines: "
    "`<REPLACE_LAST>`, the exact old final surface, and the corrected final "
    "surface. Do not emit explanation or the complete answer; the model-owned "
    "copy/edit decoder executes the script."
)


class DSET1DataError(RuntimeError):
    """The DSET1 source, edit span, split, or token custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_rows(path: Path, rows: list[dict[str, Any]]) -> str:
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


def load_bound_rows(data: Path, report: dict[str, Any], split: str) -> list[dict[str, Any]]:
    expected = report.get("outputs", {}).get(split, {})
    if (
        Path(str(expected.get("path", ""))).resolve() != data.resolve()
        or expected.get("sha256") != sha256_file(data)
    ):
        raise DSET1DataError(f"DSEO1 {split} input differs")
    rows = [json.loads(line) for line in data.read_text().splitlines() if line]
    if len(rows) != int(expected.get("rows", -1)) or any(
        row.get("schema") != DSEO1_SCHEMA for row in rows
    ):
        raise DSET1DataError(f"DSEO1 {split} rows differ")
    return rows


def registered_final_span(response: str, start: int, end: int) -> bool:
    """Require the changed span to be the structurally final answer surface."""

    markers = list(ANSWER_SENTENCE.finditer(response))
    if markers:
        marker_end = markers[-1].end()
        before = response[marker_end:start]
        after = response[end:]
        if marker_end <= start and not before.strip(" $\t") and not after.strip(" .,:;$\t"):
            return True
    boxed = boxed_inner_span(response)
    if boxed is not None:
        left, right = boxed
        suffix = response[end:right]
        if left <= start < end <= right and not any(character.isalnum() for character in suffix):
            return True
    return False


def convert_pair(pair: list[dict[str, Any]]) -> tuple[list[dict[str, Any]] | None, str | None]:
    members = {str(row["pair_member"]): row for row in pair}
    if set(members) != {"clean", "fault"} or len(pair) != 2:
        raise DSET1DataError("DSEO1 pair membership differs")
    clean, fault = members["clean"], members["fault"]
    stable_fields = (
        "pair_identity_sha256",
        "source_identity_sha256",
        "corruption_family",
        "final_response",
        "changed_character_span",
        "task",
        "training_group",
    )
    if any(clean.get(name) != fault.get(name) for name in stable_fields):
        raise DSET1DataError("DSEO1 pair stable fields differ")
    start, end = map(int, clean["changed_character_span"])
    clean_draft, fault_draft = str(clean["draft"]), str(fault["draft"])
    if not 0 <= start < end <= len(clean_draft) or len(clean_draft) != len(fault_draft):
        return None, "invalid_registered_span"
    if clean_draft[:start] != fault_draft[:start] or clean_draft[end:] != fault_draft[end:]:
        return None, "nonlocal_pair_difference"
    new, old = clean_draft[start:end], fault_draft[start:end]
    if not old or not new or old == new or "\n" in old or "\n" in new:
        return None, "invalid_edit_surface"
    if clean_draft.rfind(new) != start or fault_draft.rfind(old) != start:
        return None, "registered_span_not_last_surface"
    family = str(clean["corruption_family"])
    if not registered_final_span(clean_draft, start, end):
        return None, "registered_span_not_semantic_final"
    if str(clean["final_response"]) != clean_draft:
        return None, "clean_draft_not_final_response"
    if not str(clean["question"]).endswith(DSEO1_RESPONSE_INSTRUCTION) or not str(
        fault["question"]
    ).endswith(DSEO1_RESPONSE_INSTRUCTION):
        return None, "question_instruction_differs"

    keep_script = render_script(KEEP)
    replace_script = render_script(REPLACE_LAST, old, new)
    output = []
    for member, row, script, swapped in (
        ("clean", clean, keep_script, replace_script),
        ("fault", fault, replace_script, keep_script),
    ):
        question = str(row["question"])[: -len(DSEO1_RESPONSE_INSTRUCTION)] + INSTRUCTION
        identity = sha256_text(f"dset1\0{row['identity_sha256']}")
        output.append(
            {
                "schema": SCHEMA,
                "identity_sha256": identity,
                "pair_identity_sha256": sha256_text(
                    f"dset1-pair\0{row['pair_identity_sha256']}"
                ),
                "source_identity_sha256": row["source_identity_sha256"],
                "source_dseo1_identity_sha256": row["identity_sha256"],
                "pair_member": member,
                "corruption_family": family,
                "training_group": row["training_group"],
                "task": row["task"],
                "question": question,
                "draft": row["draft"],
                "draft_sha256": sha256_text(str(row["draft"])),
                "script": script,
                "swapped_script": swapped,
                "action": KEEP if member == "clean" else REPLACE_LAST,
                "old_surface": None if member == "clean" else old,
                "new_surface": None if member == "clean" else new,
                "gold_answer": new,
                "final_response": clean_draft,
                "changed_character_span": [start, end],
                "complete_source_retained": True,
                "complete_draft_retained": True,
                "complete_script_retained": True,
            }
        )
    return output, None


def tokenize_converted_pair(tokenizer: Any, pair_rows: list[dict[str, Any]], args: argparse.Namespace):
    maxima = Counter()
    for row in pair_rows:
        rendered = render_reasoning_messages(
            tokenizer,
            [
                {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            enable_thinking=False,
        )
        prompt = tokenizer.encode(rendered, add_special_tokens=False)
        script = tokenizer.encode(row["script"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        if len(script) > args.max_script_tokens:
            return None, "script_exceeds_frozen_decode_budget"
        total = len(prompt) + len(script)
        if total > args.max_sequence_length:
            return None, "script_presentation_overflow"
        row["prompt_token_count"] = len(prompt)
        row["script_token_count"] = len(script)
        row["total_token_count"] = total
        maxima["prompt"] = max(maxima["prompt"], len(prompt))
        maxima["script"] = max(maxima["script"], len(script))
        maxima["total"] = max(maxima["total"], total)
    return maxima, None


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise DSET1DataError("DSET1 output exists")
    dseo_report = json.loads(args.dseo_report.read_text())
    if (
        dseo_report.get("schema") != DSEO1_REPORT_SCHEMA
        or dseo_report.get("status") != "complete"
        or dseo_report.get("holdout_used") is not False
        or Path(str(dseo_report.get("source", ""))).resolve() != args.source.resolve()
        or dseo_report.get("source_sha256") != sha256_file(args.source)
        or dseo_report.get("source_report_sha256") != sha256_file(args.source_report)
        or Path(str(dseo_report.get("obr_train", ""))).resolve() != args.obr_train.resolve()
        or dseo_report.get("obr_train_sha256") != sha256_file(args.obr_train)
        or dseo_report.get("obr_report_sha256") != sha256_file(args.obr_report)
    ):
        raise DSET1DataError("DSEO1 candidate-pool binding differs")
    _bound_output(args.source_report, SOURCE_REPORT_SCHEMA, args.source)
    obr_report = _bound_output(args.obr_report, OBR1_REPORT_SCHEMA, args.obr_train)
    if obr_report.get("holdout_used") is True:
        raise DSET1DataError("DSET1 candidate pool used holdout")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    original_by_line = {}
    with args.source.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                original_by_line[line_number] = json.loads(line)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    drops = Counter()
    maxima = Counter()
    for obr in read_jsonl(args.obr_train):
        if obr.get("schema") != OBR1_SCHEMA:
            raise DSET1DataError("OBR1 candidate row differs")
        source_line = int(obr["source_line"])
        raw = original_by_line.get(source_line)
        if raw is None or str(raw.get("response", "")).strip() != str(obr["response"]).strip():
            raise DSET1DataError("DSET1 candidate source differs")
        fault = make_fault(raw, tokenizer)
        if fault is None:
            drops["no_dseo1_mutation"] += 1
            continue
        fault_response, fault_action, character_span, token_span, family, gold_answer = fault
        candidate = {
            "source_identity_sha256": str(obr["identity_sha256"]),
            "source_line": source_line,
            "training_group": str(obr["training_group"]),
            "task": str(obr["task"]),
            "raw_question": str(raw["question"]).strip(),
            "clean_response": str(raw["response"]).strip(),
            "fault_response": fault_response,
            "fault_action": fault_action,
            "corruption_family": family,
            "changed_character_span": list(character_span),
            "changed_token_span": token_span,
            "gold_answer": gold_answer,
        }
        converted, reason = convert_pair(presentations(candidate))
        if reason:
            drops[reason] += 1
            continue
        assert converted is not None
        pair_maxima, reason = tokenize_converted_pair(tokenizer, converted, args)
        if reason:
            drops[reason] += 1
            continue
        assert pair_maxima is not None
        for key, value in pair_maxima.items():
            maxima[key] = max(maxima[key], value)
        candidates[family].append(
            {
                "source_identity_sha256": candidate["source_identity_sha256"],
                "converted_rows": converted,
                "corruption_family": family,
            }
        )

    train_members, diagnostic_members, quotas = split_members(
        candidates, args.train_sources, args.diagnostic_sources
    )
    args.output.mkdir(parents=True)
    outputs = {}
    source_sets = {}
    for split, members in (("train", train_members), ("diagnostic", diagnostic_members)):
        rows = [row for member in members for row in member["converted_rows"]]
        sources = {str(row["source_identity_sha256"]) for row in rows}
        if len(rows) != 2 * len(members) or len(sources) != len(members):
            raise DSET1DataError("DSET1 selected pair cardinality differs")
        path = args.output / f"{split}.jsonl"
        digest = atomic_rows(path, rows)
        outputs[split] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "rows": len(rows),
            "sources": len(sources),
        }
        source_sets[split] = sources
    if source_sets["train"] & source_sets["diagnostic"]:
        raise DSET1DataError("DSET1 train/diagnostic source overlap")
    result = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "dseo_report": str(args.dseo_report.resolve()),
        "dseo_report_sha256": sha256_file(args.dseo_report),
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "source_report_sha256": sha256_file(args.source_report),
        "obr_train": str(args.obr_train.resolve()),
        "obr_train_sha256": sha256_file(args.obr_train),
        "obr_report_sha256": sha256_file(args.obr_report),
        "model_root": str(args.model_root.resolve()),
        "model_config_sha256": sha256_file(args.model_root / "config.json"),
        "max_sequence_length": args.max_sequence_length,
        "max_script_tokens": args.max_script_tokens,
        "complete_retention": True,
        "train_diagnostic_source_overlap": 0,
        "candidate_sources_by_family": {
            family: len(rows) for family, rows in sorted(candidates.items())
        },
        "selected_family_quotas": quotas,
        "drops": dict(drops),
        "maximum_tokens": dict(maxima),
        "outputs": outputs,
    }
    atomic_json(args.output / "report.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dseo-report", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--obr-train", type=Path, required=True)
    parser.add_argument("--obr-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--max-script-tokens", type=int, default=32)
    parser.add_argument("--train-sources", type=int, default=8192)
    parser.add_argument("--diagnostic-sources", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    report = build(parse_args())
    print(json.dumps({"outputs": report["outputs"], "drops": report["drops"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
