#!/usr/bin/env python3
"""Apply a learned Q36 independent commit scorer to three owner trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from hf_aqc1_train_commit import IndependentCommitHead, token_rows
from hf_pcf1_train_commit import hidden_states
from hf_q36_mtr_evaluate import load_q36_adapter_model, validate_adapter
from hf_q36_mtr_train_commit import (
    COMMIT_PROJECTION_CONTRACT,
    HEAD_WIDTH,
    MAX_SEQUENCE_LENGTH,
    MODEL_SCHEMA,
    Q36MTRCommitError,
    restore_commit_state,
)
from q36_mtr_roles import MODEL_REVISION, TRAINABLE_PARAMETERS

CANDIDATE_SCHEMA = "shohin-q36-mtr-model-draft-v1"
SELECTION_SCHEMA = "shohin-q36-mtr-multi-owner-selection-v1"
REPORT_SCHEMA = "shohin-q36-mtr-multi-owner-application-v1"
DEVELOPMENT_ROWS = 1_289
TASKS = ("math500", "bbh_logic", "mbpp")
LINEAGES = ("current", "owner_71", "owner_8")
SOURCE_SCHEMA = "shohin-pcf1-development-source-v1"


class Q36MTRMultiOwnerError(RuntimeError):
    """The learned multi-owner application inputs or geometry differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRMultiOwnerError(f"missing or linked input: {path}")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRMultiOwnerError(f"unreadable input: {path}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise Q36MTRMultiOwnerError(f"empty or malformed input: {path}")
    return rows


def load_development_candidates(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Load exactly one complete development trajectory per frozen identity."""

    if len(paths) != 16:
        raise Q36MTRMultiOwnerError("multi-owner candidate shard count differs")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _jsonl(path):
            if row.get("split") != "development":
                continue
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != CANDIDATE_SCHEMA
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in result
                or row.get("task") not in TASKS
                or not isinstance(row.get("completion"), str)
                or not row["completion"].strip()
                or isinstance(row.get("generated_tokens"), bool)
                or not isinstance(row.get("generated_tokens"), int)
                or row["generated_tokens"] < 0
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise Q36MTRMultiOwnerError("multi-owner candidate differs")
            result[identity] = row
    if len(result) != DEVELOPMENT_ROWS:
        raise Q36MTRMultiOwnerError("multi-owner development coverage differs")
    return result


def load_development_source(path: Path) -> dict[str, dict[str, Any]]:
    """Load the label-free problem projection used by every owner."""

    result: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != SOURCE_SCHEMA
            or row.get("split") != "development"
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in result
            or row.get("task") not in TASKS
            or not isinstance(row.get("source_prompt"), str)
            or not row["source_prompt"].strip()
            or any(field in row for field in ("assessor", "answer", "gold", "correct"))
        ):
            raise Q36MTRMultiOwnerError("multi-owner development source differs")
        result[identity] = row
    if len(result) != DEVELOPMENT_ROWS:
        raise Q36MTRMultiOwnerError("multi-owner development source coverage differs")
    return result


def choose_owner(scores: list[float]) -> int:
    """Choose the highest independent semantic score with a stable tie break."""

    if len(scores) != len(LINEAGES) or any(
        not math.isfinite(value) for value in scores
    ):
        raise Q36MTRMultiOwnerError("multi-owner semantic scores differ")
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRMultiOwnerError(f"refusing existing output: {path}")
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
        raise Q36MTRMultiOwnerError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _environment(args: argparse.Namespace) -> None:
    if sha256_file(args.environment_receipt) != args.environment_receipt_sha256:
        raise Q36MTRMultiOwnerError("multi-owner environment bytes differ")
    payload = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "shohin-q36-mtr-environment-v1"
        or payload.get("status") != "pass"
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("environment_tree_sha256") != args.environment_tree_sha256
    ):
        raise Q36MTRMultiOwnerError("multi-owner environment contract differs")


def apply(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.model_revision != MODEL_REVISION
        or args.model_loader != "causal"
        or args.max_sequence_length != MAX_SEQUENCE_LENGTH
        or args.batch_identities <= 0
        or args.output.exists()
        or args.output.is_symlink()
        or args.selections.exists()
        or args.selections.is_symlink()
        or args.report.exists()
        or args.report.is_symlink()
    ):
        raise Q36MTRMultiOwnerError("multi-owner pinned settings differ")
    _environment(args)
    source = load_development_source(args.development_source)
    owners = [
        load_development_candidates(args.current_candidates),
        load_development_candidates(args.owner71_candidates),
        load_development_candidates(args.owner8_candidates),
    ]
    identities = set(source)
    if set(owners[0]) != identities:
        raise Q36MTRMultiOwnerError("multi-owner source identities differ")
    if any(set(owner) != identities for owner in owners[1:]):
        raise Q36MTRMultiOwnerError("multi-owner identities differ")
    for identity in identities:
        tasks = {owner[identity]["task"] for owner in owners} | {
            source[identity]["task"]
        }
        if len(tasks) != 1:
            raise Q36MTRMultiOwnerError("multi-owner task binding differs")

    payload = torch.load(args.commit_checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != MODEL_SCHEMA
        or not isinstance(metadata, dict)
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("head_width") != HEAD_WIDTH
        or metadata.get("commit_projection_contract") != COMMIT_PROJECTION_CONTRACT
    ):
        raise Q36MTRMultiOwnerError("multi-owner commit checkpoint differs")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, loader = load_q36_adapter_model(
        args.model_root, args.adapter_checkpoint
    )
    if loader != "causal":
        raise Q36MTRMultiOwnerError("multi-owner model loader differs")
    trainable_receipt = validate_adapter(model, adapter_metadata, "revision")
    trainable = sorted(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if sum(parameter.numel() for _, parameter in trainable) != TRAINABLE_PARAMETERS:
        raise Q36MTRMultiOwnerError("multi-owner adapter geometry differs")
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = IndependentCommitHead(hidden_size, HEAD_WIDTH).to("cuda:0")
    try:
        restore_commit_state(trainable, head, payload)
    except Q36MTRCommitError as error:
        raise Q36MTRMultiOwnerError("multi-owner commit restore differs") from error
    model.eval()
    head.eval()
    selected_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    prompt_truncated = 0
    maximum_permutation_error = 0.0
    ordered_identities = sorted(identities)
    with torch.inference_mode():
        for start in range(0, len(ordered_identities), args.batch_identities):
            batch_identities = ordered_identities[start : start + args.batch_identities]
            encoded: list[list[int]] = []
            for identity in batch_identities:
                candidates = [owner[identity] for owner in owners]
                projected = {
                    "question": source[identity]["source_prompt"],
                    "candidates": [
                        {"completion": candidate["completion"]}
                        for candidate in candidates
                    ],
                }
                local_rows, local_truncated = token_rows(
                    tokenizer, projected, args.max_sequence_length
                )
                encoded.extend(local_rows)
                prompt_truncated += local_truncated
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = hidden_states(model, encoded, tokenizer.pad_token_id)
                grouped = hidden.reshape(-1, len(LINEAGES), hidden.shape[-1])
                direct = head.score(grouped.float()).squeeze(-1).float()
                reversed_scores = (
                    head.score(grouped.flip(1).float()).squeeze(-1).float()
                )
            maximum_permutation_error = max(
                maximum_permutation_error,
                float((direct - reversed_scores.flip(1)).abs().max().cpu()),
            )
            for identity, score_tensor in zip(
                batch_identities, direct.tolist(), strict=True
            ):
                scores = [float(value) for value in score_tensor]
                selected = choose_owner(scores)
                selected_counts[LINEAGES[selected]] += 1
                selected_rows.append(owners[selected][identity])
                ordered = sorted(scores, reverse=True)
                selection_rows.append(
                    {
                        "schema": SELECTION_SCHEMA,
                        "identity_sha256": identity,
                        "task": owners[0][identity]["task"],
                        "selected_index": selected,
                        "selected_lineage": LINEAGES[selected],
                        "scores": scores,
                        "margin": ordered[0] - ordered[1],
                        "permutation_consistent": True,
                    }
                )
    output_sha256 = _atomic_lines(args.output, selected_rows)
    selections_sha256 = _atomic_lines(args.selections, selection_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "rows": len(selected_rows),
        "lineages": list(LINEAGES),
        "selected": dict(sorted(selected_counts.items())),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "selections": str(args.selections.resolve()),
        "selections_sha256": selections_sha256,
        "commit_checkpoint_sha256": sha256_file(args.commit_checkpoint),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "prompt_truncated": prompt_truncated,
        "maximum_permutation_error": maximum_permutation_error,
        "permutation_consistent": maximum_permutation_error == 0.0,
        "commit_projection_contract": COMMIT_PROJECTION_CONTRACT,
        "task_or_correctness_visible": False,
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": args.environment_tree_sha256,
        "trainable_parameter_name_sha256": trainable_receipt[
            "trainable_parameter_name_sha256"
        ],
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--model-loader", choices=("causal",), default="causal")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--commit-checkpoint", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument(
        "--current-candidates", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--owner71-candidates", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--owner8-candidates", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--environment-tree-sha256", required=True)
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--batch-identities", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(apply(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
