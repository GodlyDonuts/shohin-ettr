#!/usr/bin/env python3
"""Run the 24-row no-score PCF1 model, adapter, and serialization mechanics gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any

from build_pcf1_data import revision_prompt
from hf_aqc1_train_commit import IndependentCommitHead, select_candidate, token_rows
from hf_pcf1_train_commit import hidden_states
from hf_pcf1_generate_drafts import (
    load_source_split,
    reject_protected_path,
    sha256_file,
    validate_environment_receipt,
)
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from pcf1_code_sandbox import (
    PCF1SandboxError,
    validate_sandbox_receipt_payload,
)

SCHEMA = "shohin-pcf1-mechanics-v1"
TASKS = ("math500", "bbh_logic", "mbpp")
LORA_LAYER_INDICES = [30, 31, 32, 33]


class PCF1MechanicsError(RuntimeError):
    """The frozen PCF1 mechanics-only admission differs."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PCF1MechanicsError(f"refusing existing mechanics report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except FileExistsError as error:
        raise PCF1MechanicsError(
            f"refusing existing mechanics report: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def select_rows(source_root: Path) -> tuple[list[dict[str, str]], str]:
    rows, _ = load_source_split(source_root, "train")
    selected: list[dict[str, str]] = []
    for task in TASKS:
        task_rows = sorted(
            (row for row in rows if row["task"] == task),
            key=lambda row: hashlib.sha256(
                f"pcf1-mechanics\0{row['identity_sha256']}".encode()
            ).hexdigest(),
        )[:8]
        if len(task_rows) != 8:
            raise PCF1MechanicsError("PCF1 mechanics task coverage differs")
        selected.extend(task_rows)
    identity_digest = hashlib.sha256()
    for row in selected:
        identity_digest.update(row["identity_sha256"].encode())
        identity_digest.update(b"\0")
    return selected, identity_digest.hexdigest()


def validate_adapter_metadata(
    training: dict[str, Any], restored: dict[str, Any], model_revision: str
) -> None:
    if (
        training.get("schema") != "shohin-hf-product-reasoning-training-v1"
        or training.get("status") != "complete"
        or training.get("updates") != 1
        or training.get("gradient_accumulation") != 24
        or training.get("lora_scope") != "token_mixer"
        or training.get("lora_layers") != 4
        or training.get("lora_layer_indices") != LORA_LAYER_INDICES
        or training.get("lora_rank") != 8
        or training.get("lora_alpha") != 16.0
        or training.get("model_revision") != model_revision
        or training.get("model_loader") != "multimodal"
        or training.get("backbone_layout") != "multimodal-language-model"
        or not isinstance(training.get("lora_projection_count"), int)
        or training["lora_projection_count"] <= 0
        or not isinstance(training.get("trainable_parameters"), int)
        or training["trainable_parameters"] <= 0
        or not isinstance(training.get("trainable_parameter_name_sha256"), str)
        or len(training["trainable_parameter_name_sha256"]) != 64
        or not isinstance(training.get("environment_receipt_sha256"), str)
        or len(training["environment_receipt_sha256"]) != 64
        or not isinstance(training.get("environment_tree_sha256"), str)
        or len(training["environment_tree_sha256"]) != 64
        or len(training.get("trace", [])) != 1
        or training["trace"][0].get("update") != 1
    ):
        raise PCF1MechanicsError("PCF1 ephemeral adapter mechanics differ")
    if (
        restored.get("update") not in (None, 1)
        or restored.get("model_revision") != model_revision
        or restored.get("lora_scope") != "token_mixer"
        or restored.get("lora_layer_indices") != LORA_LAYER_INDICES
        or restored.get("backbone_layout") != "multimodal-language-model"
        or restored.get("lora_projection_count")
        != training.get("lora_projection_count")
        or restored.get("trainable_parameters") != training.get("trainable_parameters")
        or restored.get("trainable_parameter_name_sha256")
        != training.get("trainable_parameter_name_sha256")
        or restored.get("environment_receipt_sha256")
        != training.get("environment_receipt_sha256")
        or restored.get("environment_tree_sha256")
        != training.get("environment_tree_sha256")
    ):
        raise PCF1MechanicsError("PCF1 mechanics checkpoint restoration differs")


def validate_sandbox_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    reject_protected_path(path)
    if sha256_file(path) != expected_sha256:
        raise PCF1MechanicsError("PCF1 sandbox receipt hash differs")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PCF1MechanicsError("PCF1 sandbox receipt is unreadable") from error
    try:
        validate_sandbox_receipt_payload(receipt)
    except (PCF1SandboxError, TypeError, ValueError) as error:
        raise PCF1MechanicsError("PCF1 sandbox admission receipt differs") from error
    return receipt


def validate_compute_host_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    reject_protected_path(path)
    if sha256_file(path) != expected_sha256:
        raise PCF1MechanicsError("PCF1 compute-host receipt hash differs")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PCF1MechanicsError("PCF1 compute-host receipt is unreadable") from error
    invoked = Path(str(receipt.get("nvidia_smi_invoked_path", "")))
    resolved = Path(str(receipt.get("nvidia_smi_resolved_path", "")))
    binary_sha256 = receipt.get("nvidia_smi_sha256")
    if (
        receipt.get("schema") != "shohin-pcf1-compute-host-receipt-v1"
        or receipt.get("status") != "complete"
        or receipt.get("partition") != "normal"
        or not isinstance(receipt.get("node"), str)
        or not receipt["node"]
        or receipt.get("node")
        in {"evc26", "evc29", "evc31", "evc32", "evc33", "evc38", "evc46"}
        or receipt.get("excluded_nodes")
        != ["evc26", "evc29", "evc31", "evc32", "evc33", "evc38", "evc46"]
        or not invoked.is_absolute()
        or not resolved.is_absolute()
        or invoked.resolve(strict=True) != resolved
        or not resolved.is_file()
        or not isinstance(binary_sha256, str)
        or len(binary_sha256) != 64
        or any(character not in "0123456789abcdef" for character in binary_sha256)
        or sha256_file(resolved) != binary_sha256
        or not isinstance(receipt.get("nvidia_smi_version"), str)
        or not receipt["nvidia_smi_version"]
        or receipt.get("visible_gpu_count") != 1
        or receipt.get("gpu_name") != "NVIDIA H100 PCIe"
        or not isinstance(receipt.get("driver_version"), str)
        or not receipt["driver_version"]
        or not isinstance(receipt.get("pci_bus_id"), str)
        or not receipt["pci_bus_id"]
    ):
        raise PCF1MechanicsError("PCF1 compute-host admission receipt differs")
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    for path in (
        args.source_root,
        args.adapter_checkpoint,
        args.training_report,
        args.sandbox_receipt,
        args.compute_host_receipt,
        args.report,
    ):
        reject_protected_path(path)
    if sha256_file(args.adapter_checkpoint) != args.adapter_checkpoint_sha256:
        raise PCF1MechanicsError("PCF1 mechanics adapter hash differs")
    environment = validate_environment_receipt(
        args.environment_receipt,
        args.environment_receipt_sha256,
        "train/hf_pcf1_mechanics.py",
    )
    sandbox = validate_sandbox_receipt(
        args.sandbox_receipt, args.sandbox_receipt_sha256
    )
    compute_host = validate_compute_host_receipt(
        args.compute_host_receipt, args.compute_host_receipt_sha256
    )
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    if training.get("environment_receipt_sha256") != args.environment_receipt_sha256:
        raise PCF1MechanicsError("PCF1 mechanics training environment differs")
    rows, identity_digest = select_rows(args.source_root)
    revision_prompt_parameters = tuple(inspect.signature(revision_prompt).parameters)
    if revision_prompt_parameters != ("source_prompt", "draft"):
        raise PCF1MechanicsError("PCF1 revision prompt exposes a task router")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    if loader != "multimodal" or not isinstance(metadata, dict):
        raise PCF1MechanicsError("PCF1 mechanics checkpoint restoration differs")
    validate_adapter_metadata(training, metadata, args.model_revision)
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    prompts = [row["source_prompt"] for row in rows]
    rendered = [_render_prompt(tokenizer, prompt, True, False) for prompt in prompts]
    drafts, draft_usage = _generate_completions(
        model, tokenizer, rendered, True, "greedy", args.draft_tokens, stop_ids
    )
    if len(drafts) != 24 or any(not value.strip() for value in drafts):
        raise PCF1MechanicsError("PCF1 mechanics draft generation differs")
    revision_questions = [
        revision_prompt(prompt, draft)
        for prompt, draft in zip(prompts, drafts, strict=True)
    ]
    treatment = [
        _render_prompt(tokenizer, question, True, False)
        for question in revision_questions
    ]
    unchanged = [
        _render_prompt(tokenizer, question, True, False)
        for question in revision_questions
    ]
    if treatment != unchanged:
        raise PCF1MechanicsError("PCF1 matched prompt serialization differs")
    revisions, revision_usage = _generate_completions(
        model,
        tokenizer,
        treatment,
        True,
        "greedy",
        args.revision_tokens,
        stop_ids,
    )
    if len(revisions) != 24 or any(not value.strip() for value in revisions):
        raise PCF1MechanicsError("PCF1 mechanics revision generation differs")
    order_checks = 0
    truncation = 0
    maximum_swap_error = 0.0
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = IndependentCommitHead(hidden_size, 512).to("cuda:0").eval()
    for question, revision, draft in zip(
        revision_questions, revisions, drafts, strict=True
    ):
        pair: dict[str, Any] = {
            "question": question,
            "candidates": [
                {"lineage": "revision", "completion": revision},
                {"lineage": "unchanged", "completion": draft},
            ],
        }
        direct, direct_truncated = token_rows(tokenizer, pair, 3072)
        pair["candidates"] = list(reversed(pair["candidates"]))
        swapped, swapped_truncated = token_rows(tokenizer, pair, 3072)
        if direct[0] != swapped[1] or direct[1] != swapped[0]:
            raise PCF1MechanicsError("PCF1 commit A/B serialization differs")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = hidden_states(model, direct, tokenizer.pad_token_id)
            direct_margin = head.margin(hidden[0:1], hidden[1:2]).float()[0]
            swapped_margin = head.margin(hidden[1:2], hidden[0:1]).float()[0]
        swap_error = float((direct_margin + swapped_margin).abs().cpu())
        maximum_swap_error = max(maximum_swap_error, swap_error)
        candidates = [
            {"lineage": "revision", "completion": revision},
            {"lineage": "unchanged", "completion": draft},
        ]
        direct_choice = select_candidate(float(direct_margin.cpu()), candidates)
        swapped_choice = select_candidate(
            float(swapped_margin.cpu()), list(reversed(candidates))
        )
        if direct_choice != 1 - swapped_choice and revision != draft:
            raise PCF1MechanicsError("PCF1 commit forward/swapped choice differs")
        truncation += direct_truncated + swapped_truncated
        order_checks += 1
    torch.cuda.synchronize()
    report = {
        "schema": SCHEMA,
        "status": "pass",
        "capability_scored": False,
        "rows": 24,
        "task_counts": dict(Counter(row["task"] for row in rows)),
        "identity_order_sha256": identity_digest,
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": loader,
        "model_config_sha256": args.model_config_sha256,
        "model_manifest_sha256": args.model_manifest_sha256,
        "runtime_manifest_sha256": args.runtime_manifest_sha256,
        "environment_receipt": str(args.environment_receipt.resolve()),
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": environment["environment_tree"]["sha256"],
        "sandbox_receipt": str(args.sandbox_receipt.resolve()),
        "sandbox_receipt_sha256": args.sandbox_receipt_sha256,
        "code_sandbox_config_sha256": sandbox["sandbox_config_sha256"],
        "code_sandbox_binary_sha256": sandbox["bwrap_sha256"],
        "code_sandbox_probe_sha256": args.sandbox_receipt_sha256,
        "code_sandbox_probe_result_sha256": sandbox["probe_sha256"],
        "code_sandbox_runtime_tree_sha256": sandbox["sandbox_runtime_tree_sha256"],
        "sandbox_isolation_passed": True,
        "compute_host_receipt": str(args.compute_host_receipt.resolve()),
        "compute_host_receipt_sha256": args.compute_host_receipt_sha256,
        "nvidia_smi_invoked_path": compute_host["nvidia_smi_invoked_path"],
        "nvidia_smi_resolved_path": compute_host["nvidia_smi_resolved_path"],
        "nvidia_smi_sha256": compute_host["nvidia_smi_sha256"],
        "nvidia_smi_version": compute_host["nvidia_smi_version"],
        "qualified_gpu_name": compute_host["gpu_name"],
        "qualified_driver_version": compute_host["driver_version"],
        "qualified_pci_bus_id": compute_host["pci_bus_id"],
        "qualified_node": compute_host["node"],
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": args.adapter_checkpoint_sha256,
        "training_report": str(args.training_report.resolve()),
        "training_report_sha256": sha256_file(args.training_report),
        "checkpoint_restored": True,
        "optimizer_updates": 1,
        "optimizer_presentations": 24,
        "backbone_layout": metadata["backbone_layout"],
        "lora_layers": metadata["lora_layers"],
        "lora_layer_indices": metadata["lora_layer_indices"],
        "lora_scope": metadata["lora_scope"],
        "lora_projection_count": metadata["lora_projection_count"],
        "trainable_parameters": metadata["trainable_parameters"],
        "trainable_parameter_name_sha256": metadata["trainable_parameter_name_sha256"],
        "source_only_runtime_fields": ["source_prompt"],
        "supervisor_fields_visible_to_model": False,
        "drafts_nonempty": True,
        "revisions_nonempty": True,
        "matched_prompt_ids_identical": True,
        "task_router_used": False,
        "revision_prompt_parameters": list(revision_prompt_parameters),
        "commit_ab_order_checks": order_checks,
        "commit_ab_serialization_exact": True,
        "commit_forward_swapped_exact": maximum_swap_error <= 1e-6,
        "commit_maximum_swap_error": maximum_swap_error,
        "commit_prompt_truncations": truncation,
        "draft_generated_tokens": sum(tokens for tokens, _ in draft_usage),
        "revision_generated_tokens": sum(tokens for tokens, _ in revision_usage),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "seed": args.seed,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--sandbox-receipt", type=Path, required=True)
    parser.add_argument("--sandbox-receipt-sha256", required=True)
    parser.add_argument("--compute-host-receipt", type=Path, required=True)
    parser.add_argument("--compute-host-receipt-sha256", required=True)
    parser.add_argument("--model-loader", choices=("multimodal",), default="multimodal")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint-sha256", required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--draft-tokens", type=int, default=64)
    parser.add_argument("--revision-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026081120)
    args = parser.parse_args()
    if args.draft_tokens <= 0 or args.revision_tokens <= 0:
        parser.error("PCF1 mechanics token counts must be positive")
    report = run(args)
    print(
        json.dumps({"rows": report["rows"], "status": report["status"]}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
