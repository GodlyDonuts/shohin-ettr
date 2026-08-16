#!/usr/bin/env python3
"""Train a permutation-equivariant three-owner Q36 semantic commit head."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from build_q36_mtr_setwise_commit_rows import (
    OWNER_NAMES,
    PATTERNS,
    REPORT_SCHEMA as DATA_REPORT_SCHEMA,
    ROW_SCHEMA,
)
from hf_aqc1_train_commit import token_rows
from hf_pcf1_train_commit import hidden_states
from hf_q36_mtr_evaluate import load_q36_adapter_model, validate_adapter
from hf_q36_mtr_train_commit import (
    HEAD_LR,
    HEAD_WIDTH,
    MAX_GRADIENT_NORM,
    MAX_SEQUENCE_LENGTH,
    SEED,
    TASKS,
    WEIGHT_DECAY,
    _state_sha256,
    atomic_json,
    atomic_torch,
    sha256_file,
)
from q36_mtr_roles import MODEL_REVISION
from q36_mtr_setwise_head import SetwiseCommitHead, setwise_selection_loss

MODEL_SCHEMA = "shohin-q36-mtr-setwise-commit-model-v1"
REPORT_SCHEMA = "shohin-q36-mtr-setwise-commit-training-report-v1"
TRAINING_CONSUMPTION_SCHEMA = "shohin-q36-mtr-setwise-consumption-v1"
UPDATES = 256
GRADIENT_ACCUMULATION = 8
PROJECTION = 256
BINARY_LOSS_WEIGHT = 0.5
SPLITS = ("calibration_train", "calibration_development")
PROJECTION_CONTRACT = "question_plus_three_complete_candidates_symmetric_context_v1"


class Q36MTRSetwiseTrainError(RuntimeError):
    """The setwise semantic commit training contract differs."""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRSetwiseTrainError("setwise training rows are absent or symbolic")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Q36MTRSetwiseTrainError(
                f"malformed setwise row line {number}"
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
                "correctness_pattern",
                "candidates",
            }
            or row.get("schema") != ROW_SCHEMA
            or row.get("split") not in SPLITS
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in identities
            or row.get("task") not in TASKS
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or row.get("correctness_pattern") not in PATTERNS
            or not isinstance(candidates, list)
            or len(candidates) != len(OWNER_NAMES)
            or [candidate.get("lineage") for candidate in candidates]
            != list(OWNER_NAMES)
        ):
            raise Q36MTRSetwiseTrainError("setwise training row differs")
        identities.add(identity)
        correctness: list[bool] = []
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
                or not candidate["completion"].strip()
                or not isinstance(candidate.get("correct"), bool)
                or isinstance(candidate.get("generated_tokens"), bool)
                or not isinstance(candidate.get("generated_tokens"), int)
                or candidate["generated_tokens"] < 0
                or not isinstance(candidate.get("max_token_exhausted"), bool)
            ):
                raise Q36MTRSetwiseTrainError("setwise training candidate differs")
            correctness.append(candidate["correct"])
        if (
            "".join("1" if value else "0" for value in correctness)
            != row["correctness_pattern"]
        ):
            raise Q36MTRSetwiseTrainError("setwise correctness pattern differs")
        rows.append(row)
    if not rows or {row["split"] for row in rows} != set(SPLITS):
        raise Q36MTRSetwiseTrainError("setwise calibration split coverage differs")
    return rows


def training_plan(
    rows: list[dict[str, Any]], *, seed: int, presentations: int
) -> tuple[list[int], dict[str, Any]]:
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "calibration_train":
            strata[(row["task"], row["correctness_pattern"])].append(index)
    if {pattern for _, pattern in strata} != set(PATTERNS):
        raise Q36MTRSetwiseTrainError("setwise training strata differ")
    generator = random.Random(seed)
    ordered: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for key in sorted(strata):
        generator.shuffle(strata[key])
        ordered[key] = strata[key]
    positions = dict.fromkeys(ordered, 0)
    keys = list(ordered)
    indices: list[int] = []
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    for presentation in range(presentations):
        key = keys[presentation % len(keys)]
        members = ordered[key]
        source_index = members[positions[key] % len(members)]
        positions[key] += 1
        row = rows[source_index]
        receipt = {
            "presentation": presentation,
            "source_index": source_index,
            "identity_sha256": row["identity_sha256"],
            "task": row["task"],
            "correctness_pattern": row["correctness_pattern"],
        }
        digest.update(
            (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
        indices.append(source_index)
        counts[f"{key[0]}:{key[1]}"] += 1
    return indices, {
        "schema": TRAINING_CONSUMPTION_SCHEMA,
        "seed": seed,
        "presentations": presentations,
        "unique_identities": len({rows[index]["identity_sha256"] for index in indices}),
        "presentation_sha256": digest.hexdigest(),
        "stratum_presentations": dict(sorted(counts.items())),
    }


def _token_rows(
    tokenizer: Any, row: dict[str, Any], maximum: int
) -> tuple[list[list[int]], int]:
    projected = {
        "question": row["question"],
        "candidates": [
            {"completion": candidate["completion"]} for candidate in row["candidates"]
        ],
    }
    return token_rows(tokenizer, projected, maximum)


def _metrics(
    model: Any,
    head: SetwiseCommitHead,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    maximum: int,
    batch_identities: int,
) -> tuple[dict[str, Any], int, float]:
    selected = [row for row in rows if row["split"] == "calibration_development"]
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    truncated = 0
    maximum_permutation_error = 0.0
    model.eval()
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(selected), batch_identities):
            batch = selected[start : start + batch_identities]
            encoded: list[list[int]] = []
            for row in batch:
                local, count = _token_rows(tokenizer, row, maximum)
                encoded.extend(local)
                truncated += count
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = hidden_states(model, encoded, tokenizer.pad_token_id)
                grouped = hidden.reshape(-1, len(OWNER_NAMES), hidden.shape[-1])
                scores = head(grouped).float()
                permutation = torch.tensor([2, 0, 1], device=scores.device)
                permuted = head(grouped[:, permutation]).float()
            maximum_permutation_error = max(
                maximum_permutation_error,
                float((permuted - scores[:, permutation]).abs().max().cpu()),
            )
            for row, row_scores in zip(batch, scores.tolist(), strict=True):
                choice = max(
                    range(len(row_scores)), key=lambda index: row_scores[index]
                )
                correct = [candidate["correct"] for candidate in row["candidates"]]
                for key in ("overall", row["task"]):
                    bucket = buckets[key]
                    bucket["total"] += 1
                    bucket["selected_correct"] += int(correct[choice])
                    bucket["oracle_correct"] += int(any(correct))
                    for owner, value in zip(OWNER_NAMES, correct, strict=True):
                        bucket[f"{owner}_correct"] += int(value)
                    bucket[f"selected_{OWNER_NAMES[choice]}"] += 1
    result: dict[str, Any] = {}
    for key, bucket in sorted(buckets.items()):
        total = bucket["total"]
        result[key] = {
            **dict(bucket),
            "selected_accuracy": bucket["selected_correct"] / total,
            "oracle_accuracy": bucket["oracle_correct"] / total,
        }
    return result, truncated, maximum_permutation_error


def train(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.model_revision != MODEL_REVISION
        or args.model_loader != "causal"
        or args.updates != UPDATES
        or args.gradient_accumulation != GRADIENT_ACCUMULATION
        or args.head_width != HEAD_WIDTH
        or args.projection != PROJECTION
        or args.max_sequence_length != MAX_SEQUENCE_LENGTH
        or args.head_learning_rate != HEAD_LR
        or args.binary_loss_weight != BINARY_LOSS_WEIGHT
        or args.weight_decay != WEIGHT_DECAY
        or args.max_gradient_norm != MAX_GRADIENT_NORM
        or args.seed != SEED
        or args.output.exists()
        or args.output.is_symlink()
    ):
        raise Q36MTRSetwiseTrainError("setwise pinned settings differ")
    if sha256_file(args.environment_receipt) != args.environment_receipt_sha256:
        raise Q36MTRSetwiseTrainError("setwise environment bytes differ")
    environment = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    if (
        environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("environment_tree_sha256") != args.environment_tree_sha256
    ):
        raise Q36MTRSetwiseTrainError("setwise environment contract differs")
    data_report = json.loads(args.rows_report.read_text(encoding="utf-8"))
    if (
        data_report.get("schema") != DATA_REPORT_SCHEMA
        or data_report.get("status") != "complete"
        or data_report.get("model_revision") != MODEL_REVISION
        or data_report.get("source_split") != "calibration"
        or data_report.get("rows") != 5_824
        or data_report.get("owner_lineages") != list(OWNER_NAMES)
        or data_report.get("output_sha256") != sha256_file(args.rows)
        or Path(str(data_report.get("output", ""))).resolve() != args.rows.resolve()
    ):
        raise Q36MTRSetwiseTrainError("setwise data receipt differs")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = _load_rows(args.rows)
    indices, consumption = training_plan(
        rows,
        seed=args.seed,
        presentations=args.updates * args.gradient_accumulation,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    protected_before = sha256_file(args.adapter_checkpoint)
    model, adapter_metadata, loader = load_q36_adapter_model(
        args.model_root, args.adapter_checkpoint
    )
    if loader != "causal":
        raise Q36MTRSetwiseTrainError("setwise model loader differs")
    adapter_receipt = validate_adapter(model, adapter_metadata, "revision")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = SetwiseCommitHead(hidden_size, args.head_width, args.projection).to("cuda:0")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.head_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trace: list[dict[str, Any]] = []
    truncated = 0
    update = 0
    for microstep, source_index in enumerate(indices, 1):
        row = rows[source_index]
        encoded, local_truncated = _token_rows(tokenizer, row, args.max_sequence_length)
        truncated += local_truncated
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = hidden_states(model, encoded, tokenizer.pad_token_id)
        grouped = hidden.reshape(1, len(OWNER_NAMES), hidden.shape[-1])
        correct = torch.tensor(
            [[candidate["correct"] for candidate in row["candidates"]]],
            device="cuda:0",
            dtype=torch.bool,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = head(grouped)
            loss = (
                setwise_selection_loss(
                    scores, correct, binary_weight=args.binary_loss_weight
                )
                / args.gradient_accumulation
            )
        if not torch.isfinite(loss):
            raise Q36MTRSetwiseTrainError("setwise loss is nonfinite")
        loss.backward()
        if microstep % args.gradient_accumulation:
            continue
        progress = update / max(args.updates - 1, 1)
        schedule = 0.5 * (1.0 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = args.head_learning_rate * schedule
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(head.parameters()), args.max_gradient_norm
        )
        if not torch.isfinite(gradient_norm):
            raise Q36MTRSetwiseTrainError("setwise gradient is nonfinite")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update == 1 or update % args.log_interval == 0:
            trace.append(
                {
                    "update": update,
                    "presentations": microstep,
                    "gradient_norm": float(gradient_norm),
                    "prompt_truncated": truncated,
                }
            )
    metrics, evaluation_truncated, permutation_error = _metrics(
        model,
        head,
        tokenizer,
        rows,
        args.max_sequence_length,
        args.evaluation_batch_identities,
    )
    if permutation_error != 0.0:
        raise Q36MTRSetwiseTrainError("setwise permutation equivariance differs")
    protected_after = sha256_file(args.adapter_checkpoint)
    if protected_after != protected_before:
        raise Q36MTRSetwiseTrainError("setwise source adapter changed")
    head_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in head.state_dict().items()
    }
    head_state_sha256 = _state_sha256(head_state)
    args.output.mkdir(parents=True)
    checkpoint = args.output / "setwise_commit.pt"
    metadata = {
        "model_revision": MODEL_REVISION,
        "model_loader": "causal",
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": protected_before,
        "adapter_metadata": adapter_metadata,
        "adapter_receipt": adapter_receipt,
        "rows": str(args.rows.resolve()),
        "rows_sha256": sha256_file(args.rows),
        "owner_lineages": list(OWNER_NAMES),
        "head_width": args.head_width,
        "projection": args.projection,
        "max_sequence_length": args.max_sequence_length,
        "head_learning_rate": args.head_learning_rate,
        "updates": args.updates,
        "gradient_accumulation": args.gradient_accumulation,
        "seed": args.seed,
        "binary_loss_weight": args.binary_loss_weight,
        "projection_contract": PROJECTION_CONTRACT,
        "permutation_equivariant": True,
        "backbone_frozen": True,
        "head_state_sha256": head_state_sha256,
    }
    atomic_torch(
        checkpoint,
        {"schema": MODEL_SCHEMA, "metadata": metadata, "head_state": head_state},
    )
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        restored.get("schema") != MODEL_SCHEMA
        or restored.get("metadata") != metadata
        or _state_sha256(restored.get("head_state", {})) != head_state_sha256
    ):
        raise Q36MTRSetwiseTrainError("setwise checkpoint restore differs")
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        **metadata,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_consumption": consumption,
        "calibration_development": metrics,
        "training_prompt_truncated": truncated,
        "evaluation_prompt_truncated": evaluation_truncated,
        "maximum_permutation_error": permutation_error,
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "source_adapter_unchanged": True,
        "serialization_restore_exact": True,
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": args.environment_tree_sha256,
        "trace": trace,
    }
    atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--model-loader", choices=("causal",), default="causal")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--rows-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--environment-tree-sha256", required=True)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument(
        "--gradient-accumulation", type=int, default=GRADIENT_ACCUMULATION
    )
    parser.add_argument("--head-width", type=int, default=HEAD_WIDTH)
    parser.add_argument("--projection", type=int, default=PROJECTION)
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--evaluation-batch-identities", type=int, default=2)
    parser.add_argument("--head-learning-rate", type=float, default=HEAD_LR)
    parser.add_argument("--binary-loss-weight", type=float, default=BINARY_LOSS_WEIGHT)
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
