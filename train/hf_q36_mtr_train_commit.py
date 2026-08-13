#!/usr/bin/env python3
"""Train the frozen 128-update Q36 whole-trajectory commit policy."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from build_q36_mtr_commit_pairs import (
    OUTCOMES,
    PAIR_SCHEMA,
    REPORT_SCHEMA as PAIR_REPORT_SCHEMA,
    sha256_file,
)
from hf_aqc1_train_commit import IndependentCommitHead, select_candidate, token_rows
from hf_pcf1_train_commit import (
    atomic_json,
    atomic_torch,
    evaluate,
    hidden_states,
    margins_for_batch,
)
from hf_product_reasoning_eval import _load_model
from hf_q36_mtr_evaluate import validate_adapter
from q36_mtr_roles import (
    MODEL_REVISION,
    TRAINABLE_MASTER_DTYPE,
    TRAINABLE_PARAMETERS,
)

MODEL_SCHEMA = "shohin-q36-mtr-commit-model-v1"
REPORT_SCHEMA = "shohin-q36-mtr-commit-training-report-v1"
SELECTION_SCHEMA = "shohin-q36-mtr-commit-selection-v1"
APPLICATION_SCHEMA = "shohin-q36-mtr-commit-application-report-v1"
UPDATES = 128
GRADIENT_ACCUMULATION = 8
HEAD_WIDTH = 512
MAX_SEQUENCE_LENGTH = 3072
BACKBONE_LR = 2e-6
HEAD_LR = 2e-4
TIE_LOSS_WEIGHT = 0.25
WEIGHT_DECAY = 0.01
MAX_GRADIENT_NORM = 1.0
SEED = 2026080822
TASKS = ("math500", "bbh_logic", "mbpp")
SPLITS = ("calibration_train", "calibration_development")


class Q36MTRCommitError(RuntimeError):
    """The Q36 whole-trajectory commit contract differs."""


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRCommitError("Q36 commit pairs are absent or symbolic")
    rows = []
    identities: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Q36MTRCommitError(
                f"malformed Q36 commit pair line {number}"
            ) from error
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
                "outcome_class",
                "candidates",
            }
            or row.get("schema") != PAIR_SCHEMA
            or row.get("split") not in SPLITS
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in identities
            or row.get("task") not in TASKS
            or row.get("outcome_class") not in OUTCOMES
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(candidates, list)
            or len(candidates) != 2
            or [candidate.get("lineage") for candidate in candidates]
            != ["revision", "unchanged"]
        ):
            raise Q36MTRCommitError("Q36 commit pair content differs")
        identities.add(identity)
        for candidate in candidates:
            if (
                set(candidate)
                != {
                    "lineage",
                    "completion",
                    "correct",
                    "generated_tokens",
                    "max_token_exhausted",
                }
                or not isinstance(candidate.get("completion"), str)
                or not isinstance(candidate.get("correct"), bool)
                or isinstance(candidate.get("generated_tokens"), bool)
                or not isinstance(candidate.get("generated_tokens"), int)
                or candidate["generated_tokens"] < 0
                or not isinstance(candidate.get("max_token_exhausted"), bool)
            ):
                raise Q36MTRCommitError("Q36 commit candidate content differs")
        expected = (
            "both_correct"
            if candidates[0]["correct"] and candidates[1]["correct"]
            else (
                "revision_only"
                if candidates[0]["correct"]
                else "unchanged_only" if candidates[1]["correct"] else "both_wrong"
            )
        )
        if row["outcome_class"] != expected:
            raise Q36MTRCommitError("Q36 commit outcome binding differs")
        rows.append(row)
    counts = Counter(row["split"] for row in rows)
    if len(rows) != 5_824 or set(counts) != set(SPLITS):
        raise Q36MTRCommitError("Q36 commit pair coverage differs")
    return rows


def _balanced_strata(
    rows: list[dict[str, Any]], seed: int
) -> OrderedDict[tuple[str, str], list[int]]:
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    outcomes: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if row["split"] != "calibration_train":
            continue
        key = (str(row["task"]), str(row["outcome_class"]))
        strata[key].append(index)
        outcomes[key[1]] += 1
    if set(outcomes) != set(OUTCOMES):
        raise Q36MTRCommitError("Q36 commit training lacks an outcome class")
    generator = random.Random(seed)
    result: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for key in sorted(strata):
        generator.shuffle(strata[key])
        result[key] = strata[key]
    return result


def _load_development_pairs(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRCommitError("Q36 development pairs are absent or symbolic")
    rows = []
    identities: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        identity = row.get("identity_sha256")
        candidates = row.get("candidates")
        if (
            set(row)
            != {"schema", "identity_sha256", "split", "task", "question", "candidates"}
            or row.get("schema") != PAIR_SCHEMA
            or row.get("split") != "development"
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in identities
            or row.get("task") not in TASKS
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(candidates, list)
            or len(candidates) != 2
            or [candidate.get("lineage") for candidate in candidates]
            != ["revision", "unchanged"]
            or any(
                set(candidate) != {"lineage", "completion"}
                or not isinstance(candidate.get("completion"), str)
                for candidate in candidates
            )
        ):
            raise Q36MTRCommitError("Q36 label-free development pair differs")
        identities.add(identity)
        rows.append(row)
    if len(rows) != 1_289:
        raise Q36MTRCommitError("Q36 development pair coverage differs")
    return rows


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRCommitError("Q36 commit selection output exists")
    digest = hashlib.sha256()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _validate_environment(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.environment_receipt) != args.environment_receipt_sha256:
        raise Q36MTRCommitError("Q36 commit environment bytes differ")
    environment = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    if (
        environment.get("schema") != "shohin-q36-mtr-environment-v1"
        or environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("environment_tree_sha256") != args.environment_tree_sha256
    ):
        raise Q36MTRCommitError("Q36 commit environment contract differs")
    return environment


def train(args: argparse.Namespace) -> dict[str, Any]:
    _validate_environment(args)
    from transformers import AutoTokenizer

    for path in (
        args.adapter_checkpoint,
        args.pairs,
        args.pairs_report,
        args.development_pairs,
        args.development_pairs_report,
        args.output,
        args.environment_receipt,
    ):
        rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
        if any(term in rendered for term in ("holdout", "product", "public")):
            raise Q36MTRCommitError("protected path supplied to Q36 commit")
    if (
        args.model_revision != MODEL_REVISION
        or args.model_loader != "causal"
        or args.updates != UPDATES
        or args.gradient_accumulation != GRADIENT_ACCUMULATION
        or args.head_width != HEAD_WIDTH
        or args.max_sequence_length != MAX_SEQUENCE_LENGTH
        or args.backbone_learning_rate != BACKBONE_LR
        or args.head_learning_rate != HEAD_LR
        or args.tie_loss_weight != TIE_LOSS_WEIGHT
        or args.weight_decay != WEIGHT_DECAY
        or args.max_gradient_norm != MAX_GRADIENT_NORM
        or args.seed != SEED
        or args.output.exists()
        or args.output.is_symlink()
    ):
        raise Q36MTRCommitError("Q36 pinned commit settings differ")
    pair_report = json.loads(args.pairs_report.read_text(encoding="utf-8"))
    if (
        pair_report.get("schema") != PAIR_REPORT_SCHEMA
        or pair_report.get("status") != "complete"
        or pair_report.get("model_revision") != MODEL_REVISION
        or pair_report.get("source_split") != "calibration"
        or pair_report.get("rows") != 5_824
        or Path(str(pair_report.get("output", ""))).resolve() != args.pairs.resolve()
        or pair_report.get("output_sha256") != sha256_file(args.pairs)
        or set(pair_report.get("outcomes", {})) != set(OUTCOMES)
    ):
        raise Q36MTRCommitError("Q36 commit pair receipt differs")
    development_pair_report = json.loads(
        args.development_pairs_report.read_text(encoding="utf-8")
    )
    if (
        development_pair_report.get("schema") != PAIR_REPORT_SCHEMA
        or development_pair_report.get("status") != "complete"
        or development_pair_report.get("model_revision") != MODEL_REVISION
        or development_pair_report.get("source_split") != "development"
        or development_pair_report.get("rows") != 1_289
        or development_pair_report.get("labels_or_correctness_fields") != 0
        or development_pair_report.get("source_disjoint_from_calibration") is not True
        or Path(str(development_pair_report.get("output", ""))).resolve()
        != args.development_pairs.resolve()
        or development_pair_report.get("output_sha256")
        != sha256_file(args.development_pairs)
    ):
        raise Q36MTRCommitError("Q36 development pair receipt differs")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = _load_pairs(args.pairs)
    development_rows = _load_development_pairs(args.development_pairs)
    strata = _balanced_strata(rows, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    protected_before = sha256_file(args.adapter_checkpoint)
    model, metadata, loader = _load_model(
        args.model_root,
        args.adapter_checkpoint,
        args.model_loader,
        quantization="nf4",
    )
    if loader != "causal":
        raise Q36MTRCommitError("Q36 commit model loader differs")
    trainable_receipt = validate_adapter(model, metadata, "revision")
    trainable = sorted(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if (
        sum(parameter.numel() for _, parameter in trainable) != TRAINABLE_PARAMETERS
        or any(parameter.dtype != torch.float32 for _, parameter in trainable)
        or hashlib.sha256("\n".join(name for name, _ in trainable).encode()).hexdigest()
        != trainable_receipt["trainable_parameter_name_sha256"]
    ):
        raise Q36MTRCommitError("Q36 commit trainable surface differs")
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = IndependentCommitHead(hidden_size, args.head_width).to("cuda:0")
    adapter_parameters = [parameter for _, parameter in trainable]
    head_parameters = list(head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": args.backbone_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=True,
    )
    positions = dict.fromkeys(strata, 0)
    keys = list(strata)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    head.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = microstep = presentations = truncated = 0
    trace = []
    while update < args.updates:
        key = keys[microstep % len(keys)]
        indices = strata[key]
        row = rows[indices[positions[key] % len(indices)]]
        positions[key] += 1
        encoded, local_truncated = token_rows(tokenizer, row, args.max_sequence_length)
        truncated += local_truncated
        left_correct = bool(row["candidates"][0]["correct"])
        right_correct = bool(row["candidates"][1]["correct"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            margin = margins_for_batch(model, head, encoded, tokenizer.pad_token_id)[
                0
            ].float()
            if left_correct != right_correct:
                sign = 1.0 if left_correct else -1.0
                local_loss = F.softplus(-sign * margin)
            else:
                local_loss = args.tie_loss_weight * F.smooth_l1_loss(
                    margin, torch.zeros_like(margin)
                )
            loss = local_loss / args.gradient_accumulation
        if not torch.isfinite(loss):
            raise Q36MTRCommitError("Q36 commit loss is nonfinite")
        loss.backward()
        microstep += 1
        presentations += 1
        if microstep % args.gradient_accumulation:
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            adapter_parameters + head_parameters, args.max_gradient_norm
        )
        if not torch.isfinite(gradient_norm):
            raise Q36MTRCommitError("Q36 commit gradient is nonfinite")
        progress = update / max(args.updates - 1, 1)
        schedule = 0.5 * (1.0 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = args.backbone_learning_rate * schedule
        optimizer.param_groups[1]["lr"] = args.head_learning_rate * schedule
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update == 1 or update % args.log_interval == 0:
            trace.append(
                {
                    "update": update,
                    "presentations": presentations,
                    "gradient_norm": float(gradient_norm),
                    "prompt_truncated": truncated,
                }
            )
    development, development_truncated, maximum_swap_error = evaluate(
        model,
        head,
        tokenizer,
        rows,
        "calibration_development",
        args.max_sequence_length,
        args.evaluation_batch_pairs,
    )
    protected_after = sha256_file(args.adapter_checkpoint)
    if protected_after != protected_before:
        raise Q36MTRCommitError("Q36 aligned checkpoint changed during commit training")
    args.output.mkdir(parents=True)
    checkpoint = args.output / "commit.pt"
    metadata_payload = {
        "model_revision": MODEL_REVISION,
        "model_loader": "causal",
        "quantization": "nf4",
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": protected_before,
        "adapter_metadata": metadata,
        **trainable_receipt,
        "model_root": str(args.model_source_root.resolve()),
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "head_width": args.head_width,
        "max_sequence_length": args.max_sequence_length,
        "backbone_learning_rate": args.backbone_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "updates": args.updates,
        "seed": args.seed,
        "inference_fields": ["question", "candidate_a", "candidate_b"],
        "task_or_benchmark_label_at_inference": False,
        "trainable_master_dtype": TRAINABLE_MASTER_DTYPE,
        "trainable_compute_dtype": "bfloat16",
    }
    atomic_torch(
        checkpoint,
        {
            "schema": MODEL_SCHEMA,
            "metadata": metadata_payload,
            "backbone_state": {
                name: parameter.detach().cpu() for name, parameter in trainable
            },
            "head_state": {
                name: tensor.detach().cpu()
                for name, tensor in head.state_dict().items()
            },
        },
    )
    selections: list[dict[str, Any]] = []
    application_truncated = 0
    maximum_application_swap_error = 0.0
    model.eval()
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(development_rows), args.evaluation_batch_pairs):
            batch = development_rows[start : start + args.evaluation_batch_pairs]
            encoded: list[list[int]] = []
            for row in batch:
                pair, local_truncated = token_rows(
                    tokenizer, row, args.max_sequence_length
                )
                encoded.extend(pair)
                application_truncated += local_truncated
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = hidden_states(model, encoded, tokenizer.pad_token_id)
                paired = hidden.reshape(-1, 2, hidden.shape[-1])
                direct = head.margin(paired[:, 0], paired[:, 1]).float()
                reverse = head.margin(paired[:, 1], paired[:, 0]).float()
            maximum_application_swap_error = max(
                maximum_application_swap_error,
                float((direct + reverse).abs().max().cpu()),
            )
            for row, margin, swapped_margin in zip(
                batch, direct.tolist(), reverse.tolist(), strict=True
            ):
                chosen = select_candidate(margin, row["candidates"])
                swapped = select_candidate(
                    swapped_margin, list(reversed(row["candidates"]))
                )
                consistent = chosen == 1 - swapped or (
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
                        "margin": margin,
                    }
                )
    selections_path = args.output / "development_selections.jsonl"
    selections_sha256 = _atomic_lines(selections_path, selections)
    application = {
        "schema": APPLICATION_SCHEMA,
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "commit_checkpoint": str(checkpoint.resolve()),
        "commit_checkpoint_sha256": sha256_file(checkpoint),
        "development_pairs": str(args.development_pairs.resolve()),
        "development_pairs_sha256": sha256_file(args.development_pairs),
        "development_pairs_report_sha256": sha256_file(args.development_pairs_report),
        "selections": str(selections_path.resolve()),
        "selections_sha256": selections_sha256,
        "rows": len(selections),
        "prompt_truncated": application_truncated,
        "malformed": 0,
        "order_consistent": sum(int(row["order_consistent"]) for row in selections),
        "maximum_swap_error": maximum_application_swap_error,
        "inference_fields": ["question", "candidate_a", "candidate_b"],
        "correctness_or_task_label_visible": False,
        "assessor_board_access_count": 0,
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": args.environment_tree_sha256,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    atomic_json(args.output / "application_report.json", application)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        **metadata_payload,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "development_application_report": str(
            (args.output / "application_report.json").resolve()
        ),
        "development_selections_sha256": selections_sha256,
        "calibration_development": development,
        "calibration_development_prompt_truncated": development_truncated,
        "calibration_development_maximum_swap_error": maximum_swap_error,
        "training_prompt_truncated": truncated,
        "gradient_accumulation": args.gradient_accumulation,
        "head_parameters": sum(parameter.numel() for parameter in head_parameters),
        "pair_presentations": presentations,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "protected_adapter_sha256_after": protected_after,
        "protected_adapter_unchanged": True,
        "strata_counts": {
            f"{task}:{outcome}": len(indices)
            for (task, outcome), indices in strata.items()
        },
        "trace": trace,
        "environment_verified": True,
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": args.environment_tree_sha256,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--model-loader", choices=("causal",), default="causal")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pairs-report", type=Path, required=True)
    parser.add_argument("--development-pairs", type=Path, required=True)
    parser.add_argument("--development-pairs-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--environment-tree-sha256", required=True)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument(
        "--gradient-accumulation", type=int, default=GRADIENT_ACCUMULATION
    )
    parser.add_argument("--head-width", type=int, default=HEAD_WIDTH)
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--evaluation-batch-pairs", type=int, default=2)
    parser.add_argument("--backbone-learning-rate", type=float, default=BACKBONE_LR)
    parser.add_argument("--head-learning-rate", type=float, default=HEAD_LR)
    parser.add_argument("--tie-loss-weight", type=float, default=TIE_LOSS_WEIGHT)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-gradient-norm", type=float, default=MAX_GRADIENT_NORM)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    report = train(parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "checkpoint_sha256": report["checkpoint_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
