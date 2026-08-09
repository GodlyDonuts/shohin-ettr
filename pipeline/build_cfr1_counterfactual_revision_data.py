#!/usr/bin/env python3
"""Build matched aligned/shuffled revision data from verified full solutions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from ttr1_revision import internal_revision_prompt


SCHEMA = "shohin-cfr1-counterfactual-revision-train-v1"
REPORT_SCHEMA = "shohin-cfr1-counterfactual-revision-data-report-v1"
SOURCE_REPORT_SCHEMA = "shohin-token-balanced-reasoning-mix-v1"
ADMITTED_GROUPS = ("math", "science", "code", "procedural")


class CFR1DataError(RuntimeError):
    """CFR1 source, corruption, matching, or token custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise CFR1DataError(f"refusing existing CFR1 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if path.exists() or path.is_symlink():
        raise CFR1DataError(f"refusing existing CFR1 report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def normalized_question(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().casefold()
    if not normalized:
        raise CFR1DataError("source question is empty")
    return normalized


def verification_admitted(row: dict[str, Any]) -> bool:
    group = str(row.get("training_group", ""))
    verification = str(row.get("verification", ""))
    if group in {"math", "science"}:
        return verification == "expected_answer_match_v1"
    if group == "code":
        return verification in {
            "execution_verified",
            "execution_verified_source_tests",
        }
    if group == "procedural":
        return verification == "reasoning_gym_answer_verified"
    return False


def wrong_answer(answer: str, identity: str) -> str:
    value = answer.strip()
    boxed = re.fullmatch(r"\\boxed\{(.+)\}", value, flags=re.DOTALL)
    if boxed:
        value = boxed.group(1).strip()
    if re.fullmatch(r"[-+]?\d+", value):
        offset = 1 if int(identity[:2], 16) % 2 == 0 else -1
        return str(int(value) + offset)
    fraction = re.fullmatch(r"([-+]?\d+)\s*/\s*(\d+)", value)
    if fraction:
        return f"{int(fraction.group(1)) + 1}/{fraction.group(2)}"
    if re.fullmatch(r"[A-Ja-j]", value):
        base = ord("A") if value.isupper() else ord("a")
        return chr(base + (ord(value) - base + 1) % 10)
    if value.casefold() in {"true", "false"}:
        return "false" if value.casefold() == "true" else "true"
    digest = hashlib.sha256(f"cfr1-wrong\0{identity}\0{value}".encode()).hexdigest()[:8]
    return f"incorrect_{digest}"


def counterfactual_draft(row: dict[str, Any], identity: str) -> tuple[str, str]:
    response = str(row["response"]).strip()
    group = str(row["training_group"])
    if group == "code":
        return (
            'raise RuntimeError("counterfactual revision fault")\n' + response,
            "guaranteed_runtime_failure",
        )
    answer = row.get("expected_answer_normalized") or row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise CFR1DataError("verified non-code row has no expected answer")
    wrong = wrong_answer(answer, identity)
    if wrong.casefold() == answer.strip().casefold():
        raise CFR1DataError("counterfactual answer did not change")
    return (
        response + f"\n\nA final check gives \\boxed{{{wrong}}}.",
        "contradictory_final_answer",
    )


def donor_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["training_group"])].append(row)
    donors: dict[str, str] = {}
    for group, members in by_group.items():
        if len(members) < 2:
            raise CFR1DataError(f"CFR1 donor group is singleton: {group}")
        ordered = sorted(
            members,
            key=lambda row: (len(str(row["counterfactual_draft"])), row["source_identity_sha256"]),
        )
        for index, row in enumerate(ordered):
            donor = ordered[(index + 1) % len(ordered)]
            if donor["source_identity_sha256"] == row["source_identity_sha256"]:
                raise CFR1DataError("CFR1 donor self-assignment")
            donors[str(row["source_identity_sha256"])] = str(
                donor["source_identity_sha256"]
            )
    if len(donors) != len(rows):
        raise CFR1DataError("CFR1 donor coverage differs")
    return donors


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise CFR1DataError(f"refusing existing CFR1 output root: {args.output}")
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    if (
        source_report.get("schema") != SOURCE_REPORT_SCHEMA
        or source_report.get("status") != "complete"
        or Path(str(source_report.get("output", ""))).resolve() != args.source.resolve()
        or source_report.get("output_sha256") != sha256_file(args.source)
        or int(source_report.get("max_sequence_length", -1)) != args.source_max_sequence_length
    ):
        raise CFR1DataError("CFR1 source report binding differs")

    from transformers import AutoTokenizer
    from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    seen: set[str] = set()
    canonical: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    with args.source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            group = str(raw.get("training_group", ""))
            if group not in ADMITTED_GROUPS:
                counters["excluded_group"] += 1
                continue
            if not verification_admitted(raw):
                counters[f"excluded_unverified_{group}"] += 1
                continue
            question = str(raw.get("question", "")).strip()
            response = str(raw.get("response", "")).strip()
            if not question or not response:
                counters["missing_question_or_response"] += 1
                continue
            normalized = normalized_question(question)
            source_identity = hashlib.sha256(normalized.encode()).hexdigest()
            if source_identity in seen:
                raise CFR1DataError("CFR1 source question is duplicated")
            seen.add(source_identity)
            corrupt, fault_kind = counterfactual_draft(raw, source_identity)
            canonical.append(
                {
                    "source_identity_sha256": source_identity,
                    "training_group": group,
                    "question": question,
                    "response": response,
                    "clean_draft": response,
                    "counterfactual_draft": corrupt,
                    "fault_kind": fault_kind,
                }
            )
    donors = donor_map(canonical)
    by_identity = {row["source_identity_sha256"]: row for row in canonical}

    aligned_rows: list[dict[str, Any]] = []
    shuffled_rows: list[dict[str, Any]] = []
    charged_target_tokens = 0
    prompt_token_max = target_token_max = 0
    donor_char_deltas: list[int] = []
    for row in canonical:
        donor = by_identity[donors[row["source_identity_sha256"]]]
        source_kept = True
        presentations: list[tuple[str, str, str, str]] = [
            ("verified_clean", row["clean_draft"], donor["clean_draft"], "none"),
            (
                "counterfactual_fault",
                row["counterfactual_draft"],
                donor["counterfactual_draft"],
                row["fault_kind"],
            ),
        ]
        staged: list[tuple[dict[str, Any], dict[str, Any], int]] = []
        target_ids = tokenizer.encode(row["response"], add_special_tokens=False)
        for presentation, aligned_draft, shuffled_draft, fault_kind in presentations:
            task = "mbpp" if row["training_group"] == "code" else row["training_group"]
            aligned_prompt = internal_revision_prompt(row["question"], aligned_draft, task)
            shuffled_prompt = internal_revision_prompt(row["question"], shuffled_draft, task)
            prompt_lengths = []
            for prompt in (aligned_prompt, shuffled_prompt):
                rendered = render_reasoning_messages(
                    tokenizer,
                    [
                        {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    enable_thinking=False,
                )
                prompt_lengths.append(
                    len(tokenizer.encode(rendered, add_special_tokens=False))
                )
            if any(
                prompt_length + len(target_ids) + 1 > args.max_sequence_length
                for prompt_length in prompt_lengths
            ):
                counters[f"overflow_{presentation}"] += 1
                source_kept = False
                break
            common = {
                "schema": SCHEMA,
                "source_identity_sha256": row["source_identity_sha256"],
                "training_group": row["training_group"],
                "outcome_class": presentation,
                "presentation": 0 if presentation == "verified_clean" else 1,
                "response": row["response"],
                "target_kind": "verified_full_solution",
                "fault_kind": fault_kind,
                "internal_draft_visible": True,
                "external_candidate_text_visible": False,
            }
            aligned = {
                **common,
                "identity_sha256": hashlib.sha256(
                    f"cfr1-aligned\0{row['source_identity_sha256']}\0{presentation}".encode()
                ).hexdigest(),
                "question": aligned_prompt,
                "draft_control": "aligned",
            }
            shuffled = {
                **common,
                "identity_sha256": hashlib.sha256(
                    f"cfr1-shuffled\0{row['source_identity_sha256']}\0{presentation}".encode()
                ).hexdigest(),
                "question": shuffled_prompt,
                "draft_control": "within_domain_near_length_shuffle",
                "draft_donor_identity_sha256": donor["source_identity_sha256"],
            }
            staged.append((aligned, shuffled, max(prompt_lengths)))
            donor_char_deltas.append(abs(len(aligned_draft) - len(shuffled_draft)))
        if not source_kept:
            continue
        for aligned, shuffled, prompt_tokens in staged:
            aligned_rows.append(aligned)
            shuffled_rows.append(shuffled)
            charged_target_tokens += len(target_ids) + 1
            prompt_token_max = max(prompt_token_max, prompt_tokens)
            target_token_max = max(target_token_max, len(target_ids) + 1)

    if (
        not aligned_rows
        or len(aligned_rows) != len(shuffled_rows)
        or [row["response"] for row in aligned_rows]
        != [row["response"] for row in shuffled_rows]
        or any(
            aligned["source_identity_sha256"]
            == shuffled.get("draft_donor_identity_sha256")
            for aligned, shuffled in zip(aligned_rows, shuffled_rows, strict=True)
        )
    ):
        raise CFR1DataError("CFR1 matched geometry differs")

    args.output.mkdir(parents=True)
    paths = {
        "aligned": args.output / "train_aligned.jsonl",
        "shuffled": args.output / "train_shuffled.jsonl",
    }
    hashes = {
        "aligned": atomic_lines(paths["aligned"], aligned_rows),
        "shuffled": atomic_lines(paths["shuffled"], shuffled_rows),
    }
    sorted_deltas = sorted(donor_char_deltas)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "source_report": str(args.source_report.resolve()),
        "source_report_sha256": sha256_file(args.source_report),
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "max_sequence_length": args.max_sequence_length,
        "canonical_sources_scanned": len(canonical),
        "admitted_sources": len(aligned_rows) // 2,
        "rows_per_arm": len(aligned_rows),
        "charged_target_tokens_per_arm": charged_target_tokens,
        "target_multiset_exactly_matched": True,
        "zero_source_donor_identity_matches": True,
        "counterfactual_presentations_per_arm": len(aligned_rows) // 2,
        "clean_presentations_per_arm": len(aligned_rows) // 2,
        "group_counts_per_arm": dict(
            Counter(str(row["training_group"]) for row in aligned_rows)
        ),
        "fault_counts_per_arm": dict(Counter(str(row["fault_kind"]) for row in aligned_rows)),
        "scan_counters": dict(counters),
        "prompt_token_max": prompt_token_max,
        "target_token_max": target_token_max,
        "donor_character_delta_p95": sorted_deltas[
            min(len(sorted_deltas) - 1, int(0.95 * len(sorted_deltas)))
        ],
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": hashes[name], "rows": len(aligned_rows)}
            for name, path in paths.items()
        },
        "assessor_fields_visible_to_model": False,
        "holdout_used": False,
    }
    atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-max-sequence-length", type=int, default=4096)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    report = build(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
