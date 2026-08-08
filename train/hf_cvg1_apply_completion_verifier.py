#!/usr/bin/env python3
"""Apply a qualified CVG1 verifier to matched whole-completion lineages."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from hf_cvg1_completion_verifier import (
    CompletionVerifierHead,
    MODEL_SCHEMA,
    REPORT_SCHEMA,
    bounded_token_ids,
    choose_lineage,
    configure_lora_scope,
    forward_scores,
    sha256_file,
    verifier_text,
)

PAIR_SCHEMA = "shohin-cvg1-evaluation-pairs-v1"
OUTPUT_SCHEMA = "shohin-cvg1-evaluation-selection-v1"
TASKS_BY_DOMAIN = {
    "grade_school_math": ("gsm8k",),
    "competition_math": ("math500",),
    "code": ("humaneval", "mbpp"),
    "science": ("gpqa",),
    "logic": ("bbh_logic",),
}
MAIN_TASKS = tuple(task for tasks in TASKS_BY_DOMAIN.values() for task in tasks)
TASKS = (*MAIN_TASKS, "aime")


class CVG1ApplicationError(RuntimeError):
    """The verifier or evaluation pair contract is not qualified."""


def load_evaluation_pairs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CVG1ApplicationError(f"missing evaluation pairs: {path}")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != PAIR_SCHEMA:
                raise CVG1ApplicationError(f"pair schema differs at line {line_number}")
            identity = row.get("identity_sha256")
            if not isinstance(identity, str) or len(identity) != 64:
                raise CVG1ApplicationError("pair identity is invalid")
            if identity in identities:
                raise CVG1ApplicationError("pair identity is duplicated")
            identities.add(identity)
            if row.get("task") not in TASKS:
                raise CVG1ApplicationError("pair task differs")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise CVG1ApplicationError("pair question is empty")
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 2:
                raise CVG1ApplicationError("pair must contain two candidates")
            by_lineage = {
                candidate.get("lineage"): candidate for candidate in candidates
            }
            if set(by_lineage) != {"base", "expert"}:
                raise CVG1ApplicationError("pair lineage identities differ")
            ordered = [by_lineage["base"], by_lineage["expert"]]
            for candidate in ordered:
                if (
                    not isinstance(candidate.get("completion"), str)
                    or not candidate["completion"].strip()
                ):
                    raise CVG1ApplicationError("candidate completion is empty")
                if not isinstance(candidate.get("correct"), bool):
                    raise CVG1ApplicationError("candidate correctness is invalid")
            rows.append({**row, "candidates": ordered})
    if not rows:
        raise CVG1ApplicationError("evaluation pair corpus is empty")
    if {row["task"] for row in rows} != set(TASKS):
        raise CVG1ApplicationError("evaluation pair task coverage differs")
    return rows


def _load_qualified_report(path: Path, verifier: Path, adapter: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CVG1ApplicationError(f"missing verifier report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete":
        raise CVG1ApplicationError("verifier report is incomplete")
    if report.get("holdout", {}).get("gate_pass") is not True:
        raise CVG1ApplicationError("source-disjoint verifier holdout did not pass")
    if report.get("verifier_sha256") != sha256_file(verifier):
        raise CVG1ApplicationError("verifier checkpoint hash differs")
    if report.get("adapter_checkpoint_sha256") != sha256_file(adapter):
        raise CVG1ApplicationError("protected adapter checkpoint hash differs")
    if report.get("inference_fields") != ["question", "completion"]:
        raise CVG1ApplicationError("verifier inference fields differ")
    if report.get("task_or_benchmark_label_at_inference") is not False:
        raise CVG1ApplicationError("verifier admits a task label at inference")
    return report


def _summarize(rows: list[dict[str, Any]], selected: dict[str, int]) -> dict[str, Any]:
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        candidates = row["candidates"]
        chosen = selected[row["identity_sha256"]]
        counts = task_counts[row["task"]]
        counts["total"] += 1
        counts["base_correct"] += int(candidates[0]["correct"])
        counts["expert_correct"] += int(candidates[1]["correct"])
        counts["selected_correct"] += int(candidates[chosen]["correct"])
        counts["oracle_correct"] += int(any(item["correct"] for item in candidates))
        counts["expert_commits"] += int(chosen == 1)
    tasks = {task: dict(task_counts[task]) for task in TASKS}
    arms: dict[str, dict[str, Any]] = {}
    for arm, field in (
        ("base", "base_correct"),
        ("expert", "expert_correct"),
        ("selected", "selected_correct"),
        ("oracle", "oracle_correct"),
    ):
        domains: dict[str, dict[str, Any]] = {}
        for domain, domain_tasks in TASKS_BY_DOMAIN.items():
            correct = sum(tasks[task][field] for task in domain_tasks)
            total = sum(tasks[task]["total"] for task in domain_tasks)
            domains[domain] = {
                "accuracy": correct / total,
                "correct": correct,
                "tasks": list(domain_tasks),
                "total": total,
            }
        arms[arm] = {
            "aime": {
                "accuracy": tasks["aime"][field] / tasks["aime"]["total"],
                "correct": tasks["aime"][field],
                "total": tasks["aime"]["total"],
            },
            "domains": domains,
            "macro_accuracy": sum(value["accuracy"] for value in domains.values())
            / len(domains),
            "solved": sum(tasks[task][field] for task in MAIN_TASKS),
            "total": sum(tasks[task]["total"] for task in MAIN_TASKS),
        }
    strongest_name = max(
        ("base", "expert"),
        key=lambda name: (arms[name]["macro_accuracy"], arms[name]["solved"]),
    )
    strongest = arms[strongest_name]
    treatment = arms["selected"]
    domain_deltas = {
        domain: {
            "accuracy_delta": treatment["domains"][domain]["accuracy"]
            - strongest["domains"][domain]["accuracy"],
            "solved_delta": treatment["domains"][domain]["correct"]
            - strongest["domains"][domain]["correct"],
        }
        for domain in TASKS_BY_DOMAIN
    }
    macro_delta = treatment["macro_accuracy"] - strongest["macro_accuracy"]
    solved_delta = treatment["solved"] - strongest["solved"]
    maximum_regression = max(
        max(0.0, -delta["accuracy_delta"]) for delta in domain_deltas.values()
    )
    gates = {
        "code_at_least_30_of_40": treatment["domains"]["code"]["correct"] >= 30,
        "macro_delta_at_least_three_points": macro_delta >= 0.03,
        "solved_delta_at_least_fifteen": solved_delta >= 15,
        "improves_at_least_three_domains": sum(
            delta["solved_delta"] > 0 for delta in domain_deltas.values()
        )
        >= 3,
        "no_domain_regression_over_two_points": maximum_regression <= 0.02,
    }
    return {
        "arms": arms,
        "tasks": tasks,
        "comparison": {
            "strongest_single_lineage": strongest_name,
            "domain_deltas_selected_vs_strongest": domain_deltas,
            "macro_delta_selected_vs_strongest": macro_delta,
            "solved_delta_selected_vs_strongest": solved_delta,
            "maximum_domain_regression": maximum_regression,
            "gates": gates,
            "development_gate_pass": all(gates.values()),
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise CVG1ApplicationError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def apply(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import _load_model

    rows = load_evaluation_pairs(args.pairs)
    verifier_report = _load_qualified_report(
        args.verifier_report, args.verifier, args.adapter_checkpoint
    )
    payload = torch.load(args.verifier, map_location="cpu", weights_only=True)
    if payload.get("schema") != MODEL_SCHEMA:
        raise CVG1ApplicationError("verifier checkpoint schema differs")
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("model_revision") != args.model_revision
    ):
        raise CVG1ApplicationError("verifier model metadata differs")
    if metadata.get("adapter_checkpoint_sha256") != sha256_file(
        args.adapter_checkpoint
    ):
        raise CVG1ApplicationError("verifier adapter metadata differs")

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, _, _ = _load_model(
        args.model_root, args.adapter_checkpoint, str(metadata["model_loader"])
    )
    trainable = dict(configure_lora_scope(model))
    saved_backbone = payload.get("backbone_state")
    if not isinstance(saved_backbone, dict) or set(saved_backbone) != set(trainable):
        raise CVG1ApplicationError("verifier backbone state coverage differs")
    with torch.no_grad():
        for name, parameter in trainable.items():
            saved = saved_backbone[name]
            if tuple(saved.shape) != tuple(parameter.shape):
                raise CVG1ApplicationError("verifier backbone tensor shape differs")
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = CompletionVerifierHead(hidden_size, int(metadata["head_width"])).to("cuda:0")
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    head.eval()

    started = time.monotonic()
    selected: dict[str, int] = {}
    selections: list[dict[str, Any]] = []
    truncated = 0
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_rows):
            batch = rows[start : start + args.batch_rows]
            token_rows: list[list[int]] = []
            for row in batch:
                for candidate in row["candidates"]:
                    tokens, was_truncated = bounded_token_ids(
                        tokenizer,
                        verifier_text(row["question"], candidate["completion"]),
                        int(metadata["max_sequence_length"]),
                    )
                    token_rows.append(tokens)
                    truncated += int(was_truncated)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_scores(
                    model, head, token_rows, tokenizer.pad_token_id
                ).float()
            logits_cpu = logits.cpu().tolist()
            for index, row in enumerate(batch):
                base_logit, expert_logit = map(
                    float, logits_cpu[index * 2 : index * 2 + 2]
                )
                chosen = choose_lineage(base_logit, expert_logit)
                selected[row["identity_sha256"]] = chosen
                selections.append(
                    {
                        "identity_sha256": row["identity_sha256"],
                        "base_logit": base_logit,
                        "expert_logit": expert_logit,
                        "selected_lineage": ("base", "expert")[chosen],
                        "selected_correct": row["candidates"][chosen]["correct"],
                    }
                )
            print(
                f"[cvg1-apply] {min(start + len(batch), len(rows))}/{len(rows)}",
                flush=True,
            )
    torch.cuda.synchronize()
    summary = _summarize(rows, selected)
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "verifier": str(args.verifier.resolve()),
        "verifier_sha256": sha256_file(args.verifier),
        "verifier_report": str(args.verifier_report.resolve()),
        "verifier_report_sha256": sha256_file(args.verifier_report),
        "protected_adapter": str(args.adapter_checkpoint.resolve()),
        "protected_adapter_sha256": sha256_file(args.adapter_checkpoint),
        "source_disjoint_holdout_gate_pass": verifier_report["holdout"]["gate_pass"],
        "inference_fields": ["question", "completion"],
        "task_or_benchmark_label_at_inference": False,
        "prompt_truncated": truncated,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "selections": selections,
        **summary,
        "required_next_step": (
            "open_one_source_disjoint_confirmation"
            if summary["comparison"]["development_gate_pass"]
            else "close_exact_cvg1"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--verifier-report", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=4)
    args = parser.parse_args()
    if args.batch_rows <= 0:
        parser.error("batch rows must be positive")
    report = apply(args)
    _atomic_json(args.output, report)
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
