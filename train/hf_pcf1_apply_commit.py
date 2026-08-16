#!/usr/bin/env python3
"""Apply the frozen PCF1 commit checkpoint to label-free confirmation pairs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from build_pcf1_confirmation_pairs import (
    PAIR_SCHEMA,
    REPORT_SCHEMA as PAIR_REPORT_SCHEMA,
)
from hf_aqc1_train_commit import IndependentCommitHead, select_candidate, token_rows
from hf_cvg1_completion_verifier import configure_lora_scope, sha256_file
from hf_pcf1_train_commit import MODEL_SCHEMA, hidden_states
from pcf1_environment import validate_environment_receipt

SELECTION_SCHEMA = "shohin-pcf1-commit-selection-v1"
REPORT_SCHEMA = "shohin-pcf1-commit-application-report-v1"
PINNED_MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
COMMIT_MAX_SEQUENCE_LENGTH = 3072
TASKS = ("math500", "bbh_logic", "mbpp")


class PCF1ApplyError(RuntimeError):
    """The PCF1 commit checkpoint or label-free confirmation differs."""


def reject_protected_path(path: Path) -> None:
    rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
    if any(word in rendered for word in ("holdout", "product", "public")):
        raise PCF1ApplyError(f"protected path supplied to PCF1: {path}")


def load_pairs(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise PCF1ApplyError("PCF1 pair input is missing or symbolic")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PCF1ApplyError("PCF1 pair input is unreadable") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PCF1ApplyError(f"PCF1 pair row {line_number} is malformed") from error
        if not isinstance(row, dict):
            raise PCF1ApplyError(f"PCF1 pair row {line_number} is not an object")
        identity = row.get("identity_sha256")
        candidates = row.get("candidates")
        if (
            set(row)
            != {
                "schema",
                "identity_sha256",
                "split",
                "task",
                "question",
                "candidates",
            }
            or row.get("schema") != PAIR_SCHEMA
            or row.get("split") != "confirmation"
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in identities
            or row.get("task") not in TASKS
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(candidates, list)
            or len(candidates) != 2
            or any(not isinstance(candidate, dict) for candidate in candidates)
            or [candidate.get("lineage") for candidate in candidates]
            != ["revision", "unchanged"]
        ):
            raise PCF1ApplyError("PCF1 label-free pair differs")
        identities.add(identity)
        for candidate in candidates:
            if set(candidate) != {"lineage", "completion"} or not isinstance(
                candidate.get("completion"), str
            ):
                raise PCF1ApplyError("PCF1 confirmation exposes labels")
        rows.append(row)
    if len(rows) != 1289:
        raise PCF1ApplyError("PCF1 confirmation cardinality differs")
    return rows


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise PCF1ApplyError(f"refusing existing PCF1 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    import hashlib

    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise PCF1ApplyError(f"refusing existing PCF1 output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PCF1ApplyError(f"refusing existing PCF1 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise PCF1ApplyError(f"refusing existing PCF1 output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    environment = validate_environment_receipt(
        args.environment_receipt,
        args.environment_receipt_sha256,
        "train/hf_pcf1_apply_commit.py",
    )
    from transformers import AutoTokenizer
    from hf_product_reasoning_eval import _load_model

    for path in (
        args.model_root,
        args.model_source_root,
        args.adapter_checkpoint,
        args.commit_checkpoint,
        args.pairs,
        args.pairs_report,
        args.selections,
        args.report,
        args.environment_receipt,
    ):
        reject_protected_path(path)
    for path, label in (
        (args.model_root, "staged model root"),
        (args.model_source_root, "model source root"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise PCF1ApplyError(f"PCF1 {label} is missing or symbolic")
    for path, label in (
        (args.adapter_checkpoint, "adapter checkpoint"),
        (args.commit_checkpoint, "commit checkpoint"),
        (args.pairs, "confirmation pairs"),
        (args.pairs_report, "confirmation pair report"),
        (args.environment_receipt, "environment receipt"),
    ):
        if path.is_symlink() or not path.is_file():
            raise PCF1ApplyError(f"PCF1 {label} is missing or symbolic")
    if (
        args.model_revision != PINNED_MODEL_REVISION
        or args.model_loader != "multimodal"
        or args.max_sequence_length != COMMIT_MAX_SEQUENCE_LENGTH
    ):
        raise PCF1ApplyError("PCF1 pinned application settings differ")
    if any(
        path.exists() or path.is_symlink() for path in (args.selections, args.report)
    ):
        raise PCF1ApplyError("PCF1 application output already exists")
    try:
        pair_report = json.loads(args.pairs_report.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PCF1ApplyError("PCF1 label-free pair receipt is unreadable") from error
    if (
        pair_report.get("schema") != PAIR_REPORT_SCHEMA
        or pair_report.get("status") != "complete"
        or pair_report.get("rows") != 1289
        or pair_report.get("labels_or_correctness_fields") != 0
        or pair_report.get("source_disjoint_from_calibration") is not True
        or pair_report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or Path(str(pair_report.get("output", ""))).resolve() != args.pairs.resolve()
        or pair_report.get("output_sha256") != sha256_file(args.pairs)
    ):
        raise PCF1ApplyError("PCF1 label-free pair receipt differs")
    rows = load_pairs(args.pairs)
    protected_before = sha256_file(args.adapter_checkpoint)
    payload = torch.load(args.commit_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != MODEL_SCHEMA:
        raise PCF1ApplyError("PCF1 commit checkpoint schema differs")
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("model_revision") != args.model_revision
        or metadata.get("adapter_checkpoint_sha256") != protected_before
        or metadata.get("model_loader") != "multimodal"
        or metadata.get("max_sequence_length") != COMMIT_MAX_SEQUENCE_LENGTH
        or metadata.get("head_width") != 512
        or metadata.get("model_root") != str(args.model_source_root.resolve())
    ):
        raise PCF1ApplyError("PCF1 commit checkpoint lineage differs")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    if model_loader != "multimodal":
        raise PCF1ApplyError("PCF1 multimodal loader differs")
    trainable = configure_lora_scope(model)
    state = payload.get("backbone_state")
    expected_names = [name for name, _ in trainable]
    if not isinstance(state, dict) or set(state) != set(expected_names):
        raise PCF1ApplyError("PCF1 commit backbone state differs")
    with torch.no_grad():
        for name, parameter in trainable:
            parameter.copy_(state[name].to(parameter.device, dtype=parameter.dtype))
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = IndependentCommitHead(hidden_size, int(metadata["head_width"])).to("cuda:0")
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    head.eval()
    selections: list[dict[str, Any]] = []
    truncated = 0
    maximum_swap_error = 0.0
    torch.cuda.reset_peak_memory_stats()
    import time

    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_pairs):
            batch = rows[start : start + args.batch_pairs]
            encoded: list[list[int]] = []
            for row in batch:
                pair, local_truncated = token_rows(
                    tokenizer, row, args.max_sequence_length
                )
                encoded.extend(pair)
                truncated += local_truncated
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = hidden_states(model, encoded, tokenizer.pad_token_id)
                paired = hidden.reshape(-1, 2, hidden.shape[-1])
                forward = head.margin(paired[:, 0], paired[:, 1]).float()
                swapped = head.margin(paired[:, 1], paired[:, 0]).float()
            maximum_swap_error = max(
                maximum_swap_error, float((forward + swapped).abs().max().cpu())
            )
            for row, direct, reverse in zip(
                batch, forward.tolist(), swapped.tolist(), strict=True
            ):
                chosen = select_candidate(direct, row["candidates"])
                swapped_choice = select_candidate(
                    reverse, list(reversed(row["candidates"]))
                )
                consistent = chosen == 1 - swapped_choice or (
                    row["candidates"][0]["completion"]
                    == row["candidates"][1]["completion"]
                )
                selections.append(
                    {
                        "schema": SELECTION_SCHEMA,
                        "identity_sha256": row["identity_sha256"],
                        "task": row["task"],
                        "selected_index": chosen,
                        "selected_lineage": row["candidates"][chosen]["lineage"],
                        "order_consistent": consistent,
                        "margin": direct,
                    }
                )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    protected_after = sha256_file(args.adapter_checkpoint)
    if protected_after != protected_before:
        raise PCF1ApplyError("PCF1 protected adapter changed during application")
    selections_sha256 = atomic_lines(args.selections, selections)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata": adapter_metadata,
        "commit_checkpoint": str(args.commit_checkpoint.resolve()),
        "commit_checkpoint_sha256": sha256_file(args.commit_checkpoint),
        "max_sequence_length": args.max_sequence_length,
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "pairs_report_sha256": sha256_file(args.pairs_report),
        "selections": str(args.selections.resolve()),
        "selections_sha256": selections_sha256,
        "rows": len(selections),
        "prompt_truncated": truncated,
        "malformed": 0,
        "order_consistent": sum(int(row["order_consistent"]) for row in selections),
        "maximum_swap_error": maximum_swap_error,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "inference_fields": ["question", "candidate_a", "candidate_b"],
        "correctness_or_task_label_visible": False,
        "protected_adapter_sha256_before": protected_before,
        "protected_adapter_sha256_after": protected_after,
        "protected_adapter_unchanged": protected_before == protected_after,
        "environment_verified": True,
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": environment["environment_tree"]["sha256"],
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("multimodal",), default="multimodal")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--commit-checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pairs-report", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument(
        "--max-sequence-length", type=int, default=COMMIT_MAX_SEQUENCE_LENGTH
    )
    parser.add_argument("--batch-pairs", type=int, default=2)
    args = parser.parse_args()
    if args.max_sequence_length <= 0 or args.batch_pairs <= 0:
        parser.error("PCF1 application dimensions must be positive")
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
