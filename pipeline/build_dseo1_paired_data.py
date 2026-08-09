#!/usr/bin/env python3
"""Build source-disjoint clean/fault pairs for the frozen DSEO1 canary."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

from ttr1_revision import internal_revision_prompt, tokenize_with_draft_mask


SCHEMA = "shohin-dseo1-paired-presentation-v1"
REPORT_SCHEMA = "shohin-dseo1-paired-data-report-v1"
OBR1_SCHEMA = "shohin-obr1-broad-owner-train-v1"
OBR1_REPORT_SCHEMA = "shohin-obr1-broad-owner-data-report-v1"
SOURCE_REPORT_SCHEMA = "shohin-token-balanced-reasoning-mix-v1"
ACTION_BY_FAMILY = {
    "numeric_final": "<FIX_FINAL>",
    "choice_final": "<FIX_FINAL>",
}
DSEO1_RESPONSE_INSTRUCTION = (
    "\n\nFirst emit exactly one edit action on its own line: <KEEP>, "
    "<FIX_FINAL>, <FIX_STEP>, <FIX_CODE>, or <REWRITE>. Then emit the "
    "complete final trajectory."
)
BOX_START = re.compile(r"\\boxed\s*\{")
ANSWER_SENTENCE = re.compile(r"(?i)(?:the\s+answer\s+is|answer\s*:)\s*")
SIMPLE_NUMERIC = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:/\d+)?"
)
CHOICE = re.compile(r"[A-Ea-e]")


class DSEO1DataError(RuntimeError):
    """The paired-data source, verifier, split, or retention contract differs."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_lines(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[str, int]:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    count = 0
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest(), count


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise DSEO1DataError(f"empty DSEO1 input: {path}")
    return rows


def _bound_output(report_path: Path, schema: str, data_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    expected_path = report.get("output")
    expected_sha = report.get("output_sha256")
    if report.get("schema") == OBR1_REPORT_SCHEMA:
        expected = report.get("outputs", {}).get("train", {})
        expected_path = expected.get("path", report.get("output"))
        expected_sha = expected.get("sha256", report.get("output_sha256"))
    if (
        report.get("schema") != schema
        or report.get("status") != "complete"
        or Path(str(expected_path or "")).resolve() != data_path.resolve()
        or expected_sha != sha256_file(data_path)
    ):
        raise DSEO1DataError(f"bound DSEO1 input differs: {data_path}")
    return report


def boxed_inner_span(text: str) -> tuple[int, int] | None:
    """Return the inner span of the final balanced LaTeX boxed expression."""

    matches = list(BOX_START.finditer(text))
    for match in reversed(matches):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return start, index
    return None


def final_answer_span(response: str, expected: str | None) -> tuple[int, int] | None:
    """Locate an exact, compact answer surface without rewriting reasoning."""

    boxed = boxed_inner_span(response)
    if boxed is not None:
        start, end = boxed
        content = response[start:end]
        if expected:
            occurrences = [
                match for match in re.finditer(re.escape(str(expected)), content)
            ]
            if occurrences:
                match = occurrences[-1]
                return start + match.start(), start + match.end()
        choice = list(CHOICE.finditer(content))
        if len(choice) == 1:
            return start + choice[0].start(), start + choice[0].end()
        numeric = list(SIMPLE_NUMERIC.finditer(content))
        if len(numeric) == 1:
            return start + numeric[0].start(), start + numeric[0].end()
        return None
    if expected:
        occurrences = list(re.finditer(re.escape(str(expected)), response))
        if occurrences:
            match = occurrences[-1]
            prefix = response[max(0, match.start() - 48) : match.start()]
            if ANSWER_SENTENCE.search(prefix):
                return match.start(), match.end()
    markers = list(ANSWER_SENTENCE.finditer(response))
    if not markers:
        return None
    tail_start = markers[-1].end()
    tail = response[tail_start:]
    match = SIMPLE_NUMERIC.search(tail) or CHOICE.search(tail)
    if match is None:
        return None
    return tail_start + match.start(), tail_start + match.end()


def mutate_surface(surface: str) -> str | None:
    """Produce a deterministic unequal answer with the same string width."""

    if len(surface) == 1 and CHOICE.fullmatch(surface):
        alphabet = "ABCDE" if surface.isupper() else "abcde"
        return alphabet[(alphabet.index(surface) + 1) % len(alphabet)]
    positions = [index for index, character in enumerate(surface) if character.isdigit()]
    if not positions:
        return None
    index = positions[-1]
    replacement = "1" if surface[index] == "0" else "0"
    return surface[:index] + replacement + surface[index + 1 :]


def changed_token_span(tokenizer: Any, text: str, span: tuple[int, int]) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise DSEO1DataError("OLMoE tokenizer exposes no offset mapping")
    start, end = span
    touched = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < end
    ]
    if not touched:
        raise DSEO1DataError("changed answer span maps to no token")
    return [touched[0], touched[-1] + 1]


def make_fault(
    row: dict[str, Any], tokenizer: Any
) -> tuple[str, str, tuple[int, int], list[int], str, str] | None:
    group = str(row.get("training_group", ""))
    if group not in {"procedural", "math", "science", "teacher"}:
        return None
    response = str(row.get("response", "")).strip()
    expected = row.get("answer") or row.get("expected_answer_normalized")
    span = final_answer_span(response, None if expected is None else str(expected))
    if span is None:
        return None
    surface = response[span[0] : span[1]]
    mutant = mutate_surface(surface)
    if mutant is None or mutant == surface:
        return None
    family = "choice_final" if CHOICE.fullmatch(surface) else "numeric_final"
    fault = response[: span[0]] + mutant + response[span[1] :]
    if len(fault) != len(response):
        raise DSEO1DataError("format-preserving mutation changed character width")
    token_span = changed_token_span(tokenizer, response, span)
    return fault, ACTION_BY_FAMILY[family], span, token_span, family, surface


def split_members(
    candidates: dict[str, list[dict[str, Any]]],
    train_sources: int,
    diagnostic_sources: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Select deterministic family-stratified disjoint source identities."""

    if train_sources < 4 or diagnostic_sources < 2:
        raise DSEO1DataError("DSEO1 split is too small")
    represented = sorted(family for family, rows in candidates.items() if rows)
    if len(represented) < 2:
        raise DSEO1DataError("DSEO1 requires at least two corruption families")

    # Reserve one eighth for each minority family; numeric fills the remainder.
    minority = [family for family in represented if family != "numeric_final"]
    quotas: dict[str, dict[str, int]] = {}
    train_reserved = max(1, train_sources // 8)
    diagnostic_reserved = max(1, diagnostic_sources // 8)
    for family in represented:
        quotas[family] = {
            "train": train_reserved if family in minority else 0,
            "diagnostic": diagnostic_reserved if family in minority else 0,
        }
    quotas["numeric_final"]["train"] = train_sources - sum(
        quota["train"] for family, quota in quotas.items() if family != "numeric_final"
    )
    quotas["numeric_final"]["diagnostic"] = diagnostic_sources - sum(
        quota["diagnostic"]
        for family, quota in quotas.items()
        if family != "numeric_final"
    )

    train: list[dict[str, Any]] = []
    diagnostic: list[dict[str, Any]] = []
    for family in represented:
        ordered = sorted(
            candidates[family],
            key=lambda row: sha256_text(
                f"dseo1-split\0{row['source_identity_sha256']}"
            ),
        )
        required = quotas[family]["train"] + quotas[family]["diagnostic"]
        if len(ordered) < required:
            raise DSEO1DataError(
                f"DSEO1 family {family} has {len(ordered)} rows, needs {required}"
            )
        train.extend(ordered[: quotas[family]["train"]])
        diagnostic.extend(ordered[quotas[family]["train"] : required])
    train.sort(key=lambda row: row["source_identity_sha256"])
    diagnostic.sort(key=lambda row: row["source_identity_sha256"])
    if len(train) != train_sources or len(diagnostic) != diagnostic_sources:
        raise DSEO1DataError("DSEO1 split cardinality differs")
    if {row["source_identity_sha256"] for row in train} & {
        row["source_identity_sha256"] for row in diagnostic
    }:
        raise DSEO1DataError("DSEO1 train/diagnostic source overlap")
    return train, diagnostic, quotas


def presentations(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize the clean and fault presentations for one fixed source."""

    base = {
        "schema": SCHEMA,
        "source_identity_sha256": item["source_identity_sha256"],
        "source_line": item["source_line"],
        "training_group": item["training_group"],
        "task": item["task"],
        "final_response": item["clean_response"],
        "complete_source_retained": True,
        "complete_draft_retained": True,
        "complete_target_retained": True,
        "corruption_family": item["corruption_family"],
        "changed_character_span": item["changed_character_span"],
        "changed_token_span": item["changed_token_span"],
        "clean_draft_sha256": sha256_text(item["clean_response"]),
        "fault_draft_sha256": sha256_text(item["fault_response"]),
        "clean_verifier_passed": True,
        "fault_verifier_passed": False,
        "verifier": "bound_expected_answer_inequality",
        "gold_answer": item["gold_answer"],
    }
    pair_id = sha256_text(f"dseo1-pair\0{item['source_identity_sha256']}")
    clean_action = "<KEEP>"
    fault_action = item["fault_action"]
    rows = []
    for member, draft, action, swapped in (
        ("clean", item["clean_response"], clean_action, fault_action),
        ("fault", item["fault_response"], fault_action, clean_action),
    ):
        response = f"{action}\n{item['clean_response']}"
        rows.append(
            {
                **base,
                "identity_sha256": sha256_text(f"{pair_id}\0{member}"),
                "pair_identity_sha256": pair_id,
                "pair_member": member,
                "action": action,
                "swapped_action": swapped,
                "constant_action": "<KEEP>",
                "question": (
                    internal_revision_prompt(item["raw_question"], draft, item["task"])
                    + DSEO1_RESPONSE_INSTRUCTION
                ),
                "draft": draft,
                "draft_sha256": sha256_text(draft),
                "response": response,
            }
        )
    return rows


def presentation_fits(
    tokenizer: Any,
    render_reasoning_messages: Any,
    product_system_prompt: str,
    item: dict[str, Any],
    maximum: int,
) -> bool:
    """Require both complete pair members to fit before split selection."""

    for row in presentations(item):
        rendered = render_reasoning_messages(
            tokenizer,
            [
                {"role": "system", "content": product_system_prompt},
                {"role": "user", "content": row["question"]},
            ],
            enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        response_ids = tokenizer.encode(row["response"], add_special_tokens=False)
        if len(prompt_ids) + len(response_ids) + 1 > maximum:
            return False
    return True


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise DSEO1DataError(f"refusing existing output root: {args.output}")
    source_report = _bound_output(
        args.source_report, SOURCE_REPORT_SCHEMA, args.source
    )
    obr_report = _bound_output(args.obr_report, OBR1_REPORT_SCHEMA, args.obr_train)
    if obr_report.get("holdout_used") is True:
        raise DSEO1DataError("DSEO1 source used holdout")

    from transformers import AutoTokenizer
    from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        raise DSEO1DataError("DSEO1 requires exact fast-tokenizer offsets")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    original_by_line: dict[int, dict[str, Any]] = {}
    with args.source.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                original_by_line[line_number] = json.loads(line)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    drops: Counter[str] = Counter()
    for obr in read_jsonl(args.obr_train):
        if obr.get("schema") != OBR1_SCHEMA:
            raise DSEO1DataError("OBR1 row schema differs")
        source_line = int(obr["source_line"])
        raw = original_by_line.get(source_line)
        if raw is None:
            raise DSEO1DataError("OBR1 source line is absent")
        if str(raw.get("response", "")).strip() != str(obr["response"]).strip():
            raise DSEO1DataError("OBR1 clean response differs from original source")
        fault = make_fault(raw, tokenizer)
        if fault is None:
            drops[f"no_verified_mutation_{obr['training_group']}"] += 1
            continue
        (
            fault_response,
            fault_action,
            character_span,
            token_span,
            family,
            gold_answer,
        ) = fault
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
        if not presentation_fits(
            tokenizer,
            render_reasoning_messages,
            PRODUCT_SYSTEM_PROMPT,
            candidate,
            args.max_sequence_length,
        ):
            drops[f"dseo1_overflow_{obr['training_group']}"] += 1
            continue
        candidates[family].append(candidate)

    train_sources, diagnostic_sources, quotas = split_members(
        candidates, args.train_sources, args.diagnostic_sources
    )
    args.output.mkdir(parents=True)
    outputs: dict[str, dict[str, Any]] = {}
    maxima: Counter[str] = Counter()
    action_tokens: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for split, members in (
        ("train", train_sources),
        ("diagnostic", diagnostic_sources),
    ):
        rows = [row for item in members for row in presentations(item)]
        for row in rows:
            rendered = render_reasoning_messages(
                tokenizer,
                [
                    {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                ],
                enable_thinking=False,
            )
            prompt_ids, draft_attention, _ = tokenize_with_draft_mask(tokenizer, rendered)
            response_ids = tokenizer.encode(row["response"], add_special_tokens=False) + [
                tokenizer.eos_token_id
            ]
            action_ids = tokenizer.encode(
                f"{row['action']}\n", add_special_tokens=False
            )
            final_ids = tokenizer.encode(row["final_response"], add_special_tokens=False) + [
                tokenizer.eos_token_id
            ]
            if response_ids != action_ids + final_ids:
                raise DSEO1DataError("DSEO1 action/final token concatenation differs")
            total = len(prompt_ids) + len(response_ids)
            if total > args.max_sequence_length:
                raise DSEO1DataError("selected DSEO1 row exceeds context")
            if not any(value == 0 for value in draft_attention):
                raise DSEO1DataError("selected DSEO1 draft span is absent")
            row["action_token_count"] = len(action_ids)
            row["final_token_count"] = len(final_ids)
            row["prompt_token_count"] = len(prompt_ids)
            row["draft_token_count"] = sum(1 - value for value in draft_attention)
            row["total_token_count"] = total
            action_tokens[row["action"]] += len(action_ids)
            family_counts[split][row["corruption_family"]] += 1
            maxima["prompt"] = max(maxima["prompt"], len(prompt_ids))
            maxima["draft"] = max(maxima["draft"], row["draft_token_count"])
            maxima["action"] = max(maxima["action"], len(action_ids))
            maxima["final"] = max(maxima["final"], len(final_ids))
            maxima["total"] = max(maxima["total"], total)
        path = args.output / f"{split}.jsonl"
        digest, count = atomic_lines(path, rows)
        outputs[split] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "rows": count,
            "sources": len(members),
        }

    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "source_report_sha256": sha256_file(args.source_report),
        "source_report_schema": source_report["schema"],
        "obr_train": str(args.obr_train.resolve()),
        "obr_train_sha256": sha256_file(args.obr_train),
        "obr_report_sha256": sha256_file(args.obr_report),
        "model_root": str(args.model_root.resolve()),
        "model_config_sha256": sha256_file(args.model_root / "config.json"),
        "holdout_used": False,
        "train_sources": len(train_sources),
        "diagnostic_sources": len(diagnostic_sources),
        "presentations_per_source": 2,
        "pair_balance_exact": True,
        "train_diagnostic_source_overlap": 0,
        "source_only_pair_majority_upper_bound": 0.5,
        "candidate_sources_by_family": {
            family: len(rows) for family, rows in sorted(candidates.items())
        },
        "selected_family_quotas": quotas,
        "presentation_family_counts": {
            split: dict(counts) for split, counts in family_counts.items()
        },
        "drops": dict(drops),
        "actions": sorted({"<KEEP>", *ACTION_BY_FAMILY.values()}),
        "action_token_totals": dict(action_tokens),
        "maximum_tokens": dict(maxima),
        "max_sequence_length": args.max_sequence_length,
        "complete_retention": True,
        "outputs": outputs,
    }
    atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--obr-train", type=Path, required=True)
    parser.add_argument("--obr-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-sources", type=int, default=8192)
    parser.add_argument("--diagnostic-sources", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    args = parser.parse_args()
    if min(args.train_sources, args.diagnostic_sources, args.max_sequence_length) <= 0:
        parser.error("DSEO1 dimensions must be positive")
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
