#!/usr/bin/env python3
"""Generate matched Q36 arms on a fresh external-validation source view."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from build_pcf1_data import revision_prompt
from hf_pcf1_evaluate import self_refinement_prompt, shard_bounds
from hf_product_reasoning_eval import (
    GENERATED_ONLY_SEQUENCE_CONTRACT,
    _generate_completions,
    _generation_stop_token_ids,
    _render_prompt,
)
from hf_q36_mtr_evaluate import (
    load_q36_adapter_model,
    q36_nonpadding_prompt_tokens,
    sha256_file,
    validate_adapter,
)
from q36_mtr_roles import MODEL_REVISION

SOURCE_SCHEMA = "shohin-q36-mtr-external-validation-source-v1"
CANDIDATE_SCHEMA = "shohin-q36-mtr-candidate-v1"
REPORT_SCHEMA = "shohin-q36-mtr-external-evaluation-v1"
ARMS = ("unchanged", "self_refinement", "revision", "draft_hidden", "interpolation")
TASKS = ("math500", "bbh_logic", "mbpp")
MMLU_CONFIRMATION_TASKS = ("mmlu_pro",)
MMLU_CONFIRMATION_ROWS = (256, 1_023)
ROLE_ARM = {
    "unchanged": "unchanged",
    "self_refinement": "unchanged",
    "revision": "revision",
    "draft_hidden": "draft_hidden",
    "interpolation": "revision",
}
SEED = 2026080816
SHARD_COUNTS = {256: 4, 1_023: 16, 1_279: 16}


class Q36MTRExternalEvaluationError(RuntimeError):
    """An external-validation model input or output differs."""


def adapter_validation_arm(arm: str, confirmation_mmlu_pro: bool) -> str:
    """Separate the source-only prompt arm from the frozen adapter role."""

    if arm not in ROLE_ARM:
        raise Q36MTRExternalEvaluationError("external adapter arm differs")
    if confirmation_mmlu_pro:
        if arm != "unchanged":
            raise Q36MTRExternalEvaluationError("confirmation adapter arm differs")
        return "revision"
    return ROLE_ARM[arm]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRExternalEvaluationError(f"unreadable JSONL: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise Q36MTRExternalEvaluationError(f"non-object JSONL row: {path}")
    return rows


def load_sources(
    path: Path, expected_rows: int, expected_tasks: tuple[str, ...] = TASKS
) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    identities: set[str] = set()
    for row in rows:
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != SOURCE_SCHEMA
            or row.get("split") != "external_validation"
            or row.get("task") not in expected_tasks
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in identities
            or not isinstance(row.get("source_prompt"), str)
            or not row["source_prompt"].strip()
            or row.get("runtime_fields") != ["source_prompt"]
            or any(
                field in row
                for field in ("assessor", "answer", "correct", "gold", "response")
            )
        ):
            raise Q36MTRExternalEvaluationError("external source projection differs")
        identities.add(identity)
    if len(rows) != expected_rows or {row["task"] for row in rows} != set(
        expected_tasks
    ):
        raise Q36MTRExternalEvaluationError("external source coverage differs")
    return sorted(rows, key=lambda row: row["identity_sha256"])


def load_drafts(
    paths: list[Path], sources: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    drafts: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _load_jsonl(path):
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != CANDIDATE_SCHEMA
                or row.get("arm") != "unchanged"
                or not isinstance(identity, str)
                or identity in drafts
                or not isinstance(row.get("completion"), str)
                or not row["completion"].strip()
            ):
                raise Q36MTRExternalEvaluationError("external draft differs")
            drafts[identity] = row
    identities = {row["identity_sha256"] for row in sources}
    if set(drafts) != identities:
        raise Q36MTRExternalEvaluationError("external draft coverage differs")
    return drafts


def prompt_for(arm: str, source: dict[str, Any], draft: dict[str, Any] | None) -> str:
    source_prompt = source["source_prompt"]
    if arm == "unchanged":
        if draft is not None:
            raise Q36MTRExternalEvaluationError("unchanged draft input differs")
        return source_prompt
    if draft is None or draft.get("identity_sha256") != source["identity_sha256"]:
        raise Q36MTRExternalEvaluationError("external prompt draft differs")
    completion = draft["completion"]
    if arm == "self_refinement":
        return self_refinement_prompt(
            {
                "source_prompt": source_prompt,
                "internal_draft": {"completion": completion},
            }
        )
    if arm in {"revision", "draft_hidden", "interpolation"}:
        return revision_prompt(source_prompt, completion)
    raise Q36MTRExternalEvaluationError("external arm differs")


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRExternalEvaluationError(f"refusing existing output: {path}")
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRExternalEvaluationError(f"refusing existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if (
        args.arm not in ARMS
        or args.model_revision != MODEL_REVISION
        or args.seed != SEED
        or args.expected_rows not in SHARD_COUNTS
        or args.shard_count != SHARD_COUNTS[args.expected_rows]
        or args.batch_size != 2
        or not 0 <= args.shard_index < args.shard_count
        or (
            args.confirmation_mmlu_pro
            and (
                args.arm != "unchanged"
                or args.expected_rows not in MMLU_CONFIRMATION_ROWS
            )
        )
    ):
        raise Q36MTRExternalEvaluationError("external evaluation settings differ")
    if args.candidates_output.exists() or args.report.exists():
        raise Q36MTRExternalEvaluationError("external output exists")
    if sha256_file(args.source) != args.source_sha256:
        raise Q36MTRExternalEvaluationError("external source SHA-256 differs")
    sources = load_sources(
        args.source,
        args.expected_rows,
        MMLU_CONFIRMATION_TASKS if args.confirmation_mmlu_pro else TASKS,
    )
    drafts = None
    if args.arm != "unchanged":
        if not args.draft_candidates:
            raise Q36MTRExternalEvaluationError("external drafts are absent")
        drafts = load_drafts(args.draft_candidates, sources)
    elif args.draft_candidates:
        raise Q36MTRExternalEvaluationError("unchanged must own its draft")
    row_start, row_end = shard_bounds(
        len(sources), args.shard_index, args.shard_count, args.batch_size
    )
    rows = sources[row_start:row_end]

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = load_q36_adapter_model(
        args.model_root, args.adapter_checkpoint
    )
    validation_arm = adapter_validation_arm(args.arm, args.confirmation_mmlu_pro)
    trainable_receipt = validate_adapter(model, metadata, validation_arm)
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    counters: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    started = time.monotonic()
    for source in rows:
        draft = drafts.get(source["identity_sha256"]) if drafts is not None else None
        question = prompt_for(args.arm, source, draft)
        rendered = [_render_prompt(tokenizer, question, True, False)]
        counters["prompt_tokens"] += q36_nonpadding_prompt_tokens(tokenizer, rendered)
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            768,
            stop_ids,
            add_special_tokens=False,
        )
        completion = completions[0]
        token_count, exhausted = usage[0]
        candidates.append(
            {
                "schema": CANDIDATE_SCHEMA,
                "arm": args.arm,
                "identity_sha256": source["identity_sha256"],
                "task": source["task"],
                "completion": completion,
                "generated_tokens": token_count,
                "max_token_exhausted": exhausted,
            }
        )
        counters["rows"] += 1
        counters["generated_tokens"] += token_count
        counters["max_token_exhausted"] += int(exhausted)
        counters["empty_completions"] += int(not completion.strip())
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    output_sha256 = _atomic_lines(args.candidates_output, candidates)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "split": "external_validation",
        "model_revision": MODEL_REVISION,
        "model_loader": loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata_sha256": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "adapter_validation_arm": validation_arm,
        **trainable_receipt,
        "source": str(args.source.resolve()),
        "source_sha256": args.source_sha256,
        "draft_candidate_sha256s": (
            [sha256_file(path) for path in args.draft_candidates]
            if args.draft_candidates
            else []
        ),
        "generation_mode": "greedy",
        "generation_sequence_contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
        "max_new_tokens": 768,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": row_start,
        "row_end": row_end,
        "full_row_count": len(sources),
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": output_sha256,
        "counters": dict(sorted(counters.items())),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "assessor_access_count": 0,
        "development_labels_read": 0,
        "confirmation_mmlu_pro": args.confirmation_mmlu_pro,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--draft-candidates", type=Path, action="append")
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--confirmation-mmlu-pro", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"arm": report["arm"], "rows": report["counters"]["rows"]}))


if __name__ == "__main__":
    main()
