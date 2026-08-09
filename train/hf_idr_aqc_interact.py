#!/usr/bin/env python3
"""Run the qualified model-owned draft/revise/commit reasoner."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any

from hf_aqc1_train_commit import (
    MODEL_SCHEMA,
    candidate_text,
    hidden_states,
    make_head,
    select_candidate,
)
from hf_cvg1_completion_verifier import bounded_token_ids, configure_lora_scope
from hf_idr_interact import generate_many, load_question_rows, revision_prompt
from hf_product_reasoning_eval import _load_model, _render_prompt
from package_idr_aqc_release import SCHEMA as RELEASE_SCHEMA
from package_idr_aqc_release import sha256_file
from ttr1_revision import internal_revision_prompt


REPORT_SCHEMA = "shohin-idr-aqc-interaction-v1"


class IDRAQCInteractionError(RuntimeError):
    """The release or interactive inference contract differs."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IDRAQCInteractionError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise IDRAQCInteractionError(f"{label} is not an object")
    return payload


def _read_sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise IDRAQCInteractionError("release SHA256SUMS is unreadable") from exc
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise IDRAQCInteractionError("release SHA256SUMS is malformed")
        digest, name = parts
        if (
            name in entries
            or Path(name).name != name
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise IDRAQCInteractionError("release SHA256SUMS entry is invalid")
        entries[name] = digest
    if not entries:
        raise IDRAQCInteractionError("release SHA256SUMS is empty")
    return entries


def verify_release(release_root: Path, model_root: Path) -> dict[str, Any]:
    if not release_root.is_dir() or not model_root.is_dir():
        raise IDRAQCInteractionError("release or model root is missing")
    sums = _read_sums(release_root / "SHA256SUMS")
    actual_files = {
        path.name
        for path in release_root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(sums) != actual_files:
        raise IDRAQCInteractionError("release file coverage differs")
    for name, expected in sums.items():
        if sha256_file(release_root / name) != expected:
            raise IDRAQCInteractionError(f"release file SHA-256 differs: {name}")

    manifest = _load_json(release_root / "manifest.json", "release manifest")
    if manifest.get("schema") != RELEASE_SCHEMA or manifest.get("status") != "qualified":
        raise IDRAQCInteractionError("release is not qualified")
    expected_stages = [
        "internal_draft",
        "trained_revision",
        "unchanged_continuation",
        "whole_trajectory_commit",
    ]
    if manifest.get("inference_stages") != expected_stages:
        raise IDRAQCInteractionError("release inference stages differ")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise IDRAQCInteractionError("release manifest file map is missing")
    for name, expected in manifest_files.items():
        if sums.get(name) != expected:
            raise IDRAQCInteractionError("release manifest file binding differs")
    config = model_root / "config.json"
    if not config.is_file() or sha256_file(config) != manifest.get(
        "model_config_sha256"
    ):
        raise IDRAQCInteractionError("base model config binding differs")
    return manifest


def exact_revision_prompt(question: str, draft: str, response_mode: str) -> str:
    """Use the trained prompt for qualified modes and explicit general fallback."""
    if response_mode == "code":
        return internal_revision_prompt(question, draft, "mbpp")
    if response_mode == "math":
        return internal_revision_prompt(question, draft, "math500")
    if response_mode == "general":
        return revision_prompt(question, draft, response_mode)
    raise IDRAQCInteractionError("unsupported response mode")


def _generation_receipt(
    tokenizer: Any,
    prompts: list[str],
    usage: list[dict[str, Any]],
    elapsed_seconds: float,
    peak_gpu_memory_bytes: int,
) -> dict[str, Any]:
    rendered = [_render_prompt(tokenizer, prompt, True, False) for prompt in prompts]
    prompt_tokens = [
        len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in rendered
    ]
    return {
        "rows": len(prompts),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": [int(item["generated_tokens"]) for item in usage],
        "max_token_exhausted": [bool(item["max_token_exhausted"]) for item in usage],
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
    }


def generate_stage(
    model_root: Path,
    checkpoint: Path,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    seed: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    import torch

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    outputs, usage, metadata = generate_many(
        model_root, checkpoint, tokenizer, prompts, max_new_tokens, seed
    )
    elapsed = time.monotonic() - started
    receipt = _generation_receipt(
        tokenizer,
        prompts,
        usage,
        elapsed,
        int(torch.cuda.max_memory_allocated()),
    )
    return outputs, usage, metadata, receipt


def validate_commit_payload(
    payload: dict[str, Any],
    report: dict[str, Any],
    draft_checkpoint_sha256: str,
    commit_checkpoint_sha256: str,
) -> dict[str, Any]:
    if payload.get("schema") != MODEL_SCHEMA:
        raise IDRAQCInteractionError("commit checkpoint schema differs")
    if report.get("schema") != "shohin-aqc1-commit-report-v1":
        raise IDRAQCInteractionError("commit report schema differs")
    if report.get("status") != "complete" or report.get("holdout_gate_pass") is not True:
        raise IDRAQCInteractionError("commit report is not qualified")
    if report.get("arm") != "antisymmetric":
        raise IDRAQCInteractionError("commit arm differs")
    if report.get("checkpoint_sha256") != commit_checkpoint_sha256:
        raise IDRAQCInteractionError("commit checkpoint/report binding differs")
    if report.get("adapter_checkpoint_sha256") != draft_checkpoint_sha256:
        raise IDRAQCInteractionError("commit draft-adapter binding differs")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise IDRAQCInteractionError("commit metadata is missing")
    if metadata.get("arm") != "antisymmetric":
        raise IDRAQCInteractionError("commit metadata arm differs")
    if metadata.get("adapter_checkpoint_sha256") != draft_checkpoint_sha256:
        raise IDRAQCInteractionError("commit metadata adapter binding differs")
    if not isinstance(payload.get("backbone_state"), dict) or not isinstance(
        payload.get("head_state"), dict
    ):
        raise IDRAQCInteractionError("commit state is incomplete")
    return metadata


def commit_trajectories(
    model_root: Path,
    draft_checkpoint: Path,
    commit_checkpoint: Path,
    commit_report_path: Path,
    tokenizer: Any,
    questions: list[str],
    revisions: list[str],
    controls: list[str],
    batch_pairs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    if not (len(questions) == len(revisions) == len(controls)):
        raise IDRAQCInteractionError("commit candidate cardinality differs")
    commit_sha = sha256_file(commit_checkpoint)
    draft_sha = sha256_file(draft_checkpoint)
    report = _load_json(commit_report_path, "commit report")
    payload = torch.load(commit_checkpoint, map_location="cpu", weights_only=True)
    metadata = validate_commit_payload(payload, report, draft_sha, commit_sha)

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model, adapter_metadata, model_loader = _load_model(
        model_root, draft_checkpoint, "multimodal"
    )
    trainable = dict(configure_lora_scope(model))
    if set(trainable) != set(payload["backbone_state"]):
        raise IDRAQCInteractionError("commit backbone-state coverage differs")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(
                payload["backbone_state"][name].to(parameter.device, parameter.dtype)
            )
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = make_head(
        "antisymmetric",
        hidden_size,
        int(metadata["head_width"]),
        int(metadata["projection_width"]),
    ).to("cuda:0")
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    head.eval()

    rows = [
        {
            "question": question,
            "candidates": [
                {"lineage": "trained_revision", "completion": revision},
                {"lineage": "unchanged_continuation", "completion": control},
            ],
        }
        for question, revision, control in zip(
            questions, revisions, controls, strict=True
        )
    ]
    outputs: list[dict[str, Any]] = []
    maximum_swap_error = 0.0
    maximum = int(metadata["max_sequence_length"])
    with torch.inference_mode():
        for start in range(0, len(rows), batch_pairs):
            batch = rows[start : start + batch_pairs]
            encoded: list[list[int]] = []
            for row in batch:
                for candidate in row["candidates"]:
                    tokens, truncated = bounded_token_ids(
                        tokenizer,
                        candidate_text(row["question"], candidate["completion"]),
                        maximum,
                    )
                    if truncated:
                        raise IDRAQCInteractionError(
                            "commit candidate exceeds qualified context"
                        )
                    encoded.append(tokens)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = hidden_states(model, encoded, tokenizer.pad_token_id)
                paired = hidden.reshape(-1, 2, hidden.shape[-1])
                direct = head.margin(paired[:, 0], paired[:, 1]).float()
                reverse = head.margin(paired[:, 1], paired[:, 0]).float()
            maximum_swap_error = max(
                maximum_swap_error, float((direct + reverse).abs().max().cpu())
            )
            for row, margin, reverse_margin in zip(
                batch, direct.tolist(), reverse.tolist(), strict=True
            ):
                selected = select_candidate(margin, row["candidates"])
                reverse_selected = select_candidate(
                    reverse_margin, list(reversed(row["candidates"]))
                )
                consistent = selected == 1 - reverse_selected or (
                    row["candidates"][0]["completion"]
                    == row["candidates"][1]["completion"]
                )
                if not consistent:
                    raise IDRAQCInteractionError("commit order consistency failed")
                candidate = row["candidates"][selected]
                outputs.append(
                    {
                        "selected_index": selected,
                        "selected_lineage": candidate["lineage"],
                        "selected_answer": candidate["completion"],
                        "margin": margin,
                        "reverse_margin": reverse_margin,
                        "order_consistent": True,
                    }
                )
    receipt = {
        "rows": len(rows),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "maximum_swap_error": maximum_swap_error,
        "model_loader": model_loader,
        "adapter_metadata": adapter_metadata,
        "commit_checkpoint_sha256": commit_sha,
        "commit_report_sha256": sha256_file(commit_report_path),
        "candidate_context_limit": maximum,
        "candidate_truncated": 0,
    }
    del model, head
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, receipt


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise IDRAQCInteractionError("refusing to replace interaction report")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    manifest = verify_release(args.release_root, args.model_root)
    rows = load_question_rows(args.questions_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    draft_checkpoint = args.release_root / "draft_adapter.pt"
    revision_checkpoint = args.release_root / "revision_adapter.pt"
    commit_checkpoint = args.release_root / "commit.pt"
    commit_report = args.release_root / "commit_report.json"

    questions = [row["question"] for row in rows]
    drafts, draft_usage, draft_metadata, draft_receipt = generate_stage(
        args.model_root,
        draft_checkpoint,
        tokenizer,
        questions,
        args.max_new_tokens,
        args.seed,
    )
    revision_prompts = [
        exact_revision_prompt(row["question"], draft, row["response_mode"])
        for row, draft in zip(rows, drafts, strict=True)
    ]
    revisions, revision_usage, revision_metadata, revision_receipt = generate_stage(
        args.model_root,
        revision_checkpoint,
        tokenizer,
        revision_prompts,
        args.max_new_tokens,
        args.seed + len(rows),
    )
    controls, control_usage, control_metadata, control_receipt = generate_stage(
        args.model_root,
        draft_checkpoint,
        tokenizer,
        revision_prompts,
        args.max_new_tokens,
        args.seed + 2 * len(rows),
    )
    committed, commit_receipt = commit_trajectories(
        args.model_root,
        draft_checkpoint,
        commit_checkpoint,
        commit_report,
        tokenizer,
        questions,
        revisions,
        controls,
        args.batch_pairs,
    )
    interactions = []
    for row, draft, revision, control, chosen, draft_use, revision_use, control_use in zip(
        rows,
        drafts,
        revisions,
        controls,
        committed,
        draft_usage,
        revision_usage,
        control_usage,
        strict=True,
    ):
        interactions.append(
            {
                **row,
                "internal_draft": draft,
                "trained_revision": revision,
                "unchanged_continuation": control,
                **chosen,
                "generation": {
                    "draft": draft_use,
                    "revision": revision_use,
                    "control": control_use,
                },
            }
        )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "release_root": str(args.release_root.resolve()),
        "release_manifest_sha256": sha256_file(args.release_root / "manifest.json"),
        "release_sums_sha256": sha256_file(args.release_root / "SHA256SUMS"),
        "model_root": str(args.model_root.resolve()),
        "model_revision": manifest["model_revision"],
        "questions": str(args.questions_jsonl.resolve()),
        "questions_sha256": sha256_file(args.questions_jsonl),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "runtime_fields": ["question", "response_mode"],
        "external_models_or_tools": False,
        "stages": {
            "draft": {**draft_receipt, "model": draft_metadata},
            "revision": {**revision_receipt, "model": revision_metadata},
            "control": {**control_receipt, "model": control_metadata},
            "commit": commit_receipt,
        },
        "interactions": interactions,
    }
    _atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--questions-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=2026080907)
    parser.add_argument("--batch-pairs", type=int, default=2)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.batch_pairs <= 0:
        parser.error("token and batch limits must be positive")
    report = run(args)
    for interaction in report["interactions"]:
        print(f"=== {interaction['id']} ({interaction['selected_lineage']}) ===")
        print(interaction["selected_answer"])
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
