"""Memory-bounded on-policy verified-reward training for the product reasoner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from hf_product_reasoning_eval import (
    _completion_usage,
    _generate_adapter,
    _generation_arguments,
    _generation_stop_token_ids,
    _render_prompt,
)
from hf_product_reasoning_rollouts import score_completion
from hf_product_reasoning_train import (
    ProductReasoningModel,
    _atomic_json,
    _save_checkpoint,
    _sha256_file,
    _tokenize_rows,
    load_product_backbone,
    load_trainable_checkpoint,
    reservoir_rows_with_sha256,
    validate_warm_start_metadata,
)


SCHEMA = "shohin-hf-product-reasoning-rlvr-training-v1"


class ProductRLVRTrainError(RuntimeError):
    """On-policy training cannot satisfy its model, data, or reward contract."""


def standardized_group_advantages(
    rewards: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Center binary rewards within one prompt and suppress uniform groups."""

    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ProductRLVRTrainError("a reward group must contain at least two values")
    centered = rewards.float() - rewards.float().mean()
    scale = centered.square().mean().sqrt()
    if float(scale) <= epsilon:
        return torch.zeros_like(centered)
    return centered / scale


def policy_objective(logp: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    """One-use on-policy sequence objective; no stale importance ratio is needed."""

    return -(advantage.detach() * logp)


def verified_terminal_reward(candidate: dict[str, Any]) -> float:
    """Reward only an exact answer emitted by a self-terminated trajectory."""

    return float(candidate["correct"] and not candidate["max_token_exhausted"])


def _average_logp(
    model: ProductReasoningModel,
    prompt_ids: list[int],
    response_ids: list[int],
    pad_token_id: int,
) -> tuple[torch.Tensor, int]:
    loss, metrics = model.forward_batch([prompt_ids], [response_ids], pad_token_id)
    return -loss, int(metrics["charged_tokens"])


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ProductRLVRTrainError(f"refusing to replace rollout ledger: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _reservoir_reward_rows(
    path: Path,
    limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], str]:
    """Hash and sample verifier rows without discarding identity or gold fields."""

    if limit <= 0:
        raise ProductRLVRTrainError("reward row limit must be positive")
    generator = random.Random(seed)
    digest = hashlib.sha256()
    selected: list[dict[str, Any]] = []
    valid = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ProductRLVRTrainError("reward bank is malformed") from exc
            if not isinstance(row, dict):
                raise ProductRLVRTrainError("reward-bank row is not an object")
            valid += 1
            if len(selected) < limit:
                selected.append(row)
            else:
                position = generator.randrange(valid)
                if position < limit:
                    selected[position] = row
    if not selected:
        raise ProductRLVRTrainError("reward bank has no rows")
    generator.shuffle(selected)
    return selected, digest.hexdigest()


def _validate_reward_rows(rows: list[dict[str, Any]]) -> None:
    identities: set[str] = set()
    for row in rows:
        identity = str(row.get("identity_sha256") or "")
        if not identity or identity in identities:
            raise ProductRLVRTrainError(
                "reward-bank identities are empty or duplicated"
            )
        identities.add(identity)
        if row.get("task") != "math500":
            raise ProductRLVRTrainError("the first RLVR pilot requires MATH rows")
        if not str(row.get("question") or "").strip():
            raise ProductRLVRTrainError("reward-bank question is empty")
        if row.get("answer") is None:
            raise ProductRLVRTrainError("reward-bank answer is missing")


def _validate_resume_contract(
    warm_update: int,
    warm_metadata: dict[str, Any],
    args: argparse.Namespace,
    reward_data_sha256: str,
    replay_data_sha256: str,
) -> None:
    """Require every short backfill chunk to continue one exact RLVR trajectory."""

    if args.start_update == 0:
        if warm_metadata.get("rlvr_algorithm") is not None:
            raise ProductRLVRTrainError("a zero-offset run cannot resume an RLVR chunk")
        return
    expected = {
        "rlvr_algorithm": "single_use_on_policy_group_normalized_reinforce_v1",
        "rlvr_reward": "exact_math_answer_with_explicit_marker_and_terminal_stop_v1",
        "data_sha256": reward_data_sha256,
        "rlvr_replay_data_sha256": replay_data_sha256,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "rlvr_replay_data_seed": args.replay_data_seed,
        "rlvr_samples": args.samples,
        "rlvr_groups_per_update": args.groups_per_update,
        "rlvr_max_new_tokens": args.max_new_tokens,
        "rlvr_replay_weight": args.replay_weight,
        "rlvr_schedule_total_updates": args.schedule_total_updates,
    }
    mismatches = {
        key: {"expected": value, "actual": warm_metadata.get(key)}
        for key, value in expected.items()
        if warm_metadata.get(key) != value
    }
    if warm_update != args.start_update:
        mismatches["checkpoint_update"] = {
            "expected": args.start_update,
            "actual": warm_update,
        }
    if mismatches:
        raise ProductRLVRTrainError(
            f"RLVR resume contract differs: {json.dumps(mismatches, sort_keys=True)}"
        )


def _restore_optimizer(path: Path, optimizer: torch.optim.Optimizer) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    optimizer_state = payload.get("optimizer")
    if not isinstance(optimizer_state, dict):
        raise ProductRLVRTrainError("RLVR checkpoint optimizer state is missing")
    optimizer.load_state_dict(optimizer_state)


def _generate_group_batch(
    model: ProductReasoningModel,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    samples: int,
    max_new_tokens: int,
    seed: int,
    stop_token_ids: list[int],
) -> list[list[dict[str, Any]]]:
    rendered_prompts = [
        _render_prompt(tokenizer, str(row["question"]), True, False) for row in rows
    ]
    rendered = [prompt for prompt in rendered_prompts for _ in range(samples)]
    encoded = tokenizer(rendered, padding=True, return_tensors="pt")
    encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    arguments = _generation_arguments("qwen-thinking", max_new_tokens)
    arguments["eos_token_id"] = (
        stop_token_ids[0] if len(stop_token_ids) == 1 else stop_token_ids
    )
    model.eval()
    with torch.inference_mode():
        completion_ids = _generate_adapter(
            model,
            encoded,
            arguments,
            tokenizer.pad_token_id,
        )
    model.train()

    groups: list[list[dict[str, Any]]] = []
    for row_index, (row, rendered_prompt) in enumerate(
        zip(rows, rendered_prompts, strict=True)
    ):
        prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
        candidates: list[dict[str, Any]] = []
        for sample_index in range(samples):
            flat_index = row_index * samples + sample_index
            raw_ids = completion_ids[flat_index].tolist()
            token_count, exhausted = _completion_usage(
                raw_ids, stop_token_ids, max_new_tokens
            )
            response_ids = raw_ids[:token_count]
            completion = tokenizer.decode(response_ids, skip_special_tokens=True)
            score = score_completion(row, completion)
            candidates.append(
                {
                    "sample_index": sample_index,
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "completion": completion,
                    "generated_tokens": token_count,
                    "max_token_exhausted": exhausted,
                    **score,
                }
            )
        groups.append(candidates)
    return groups


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.arm != "baseline":
        raise ProductRLVRTrainError("the first RLVR pilot requires baseline arm")
    if args.output.exists():
        raise ProductRLVRTrainError(f"output already exists: {args.output}")
    if not args.warm_start_checkpoint.is_file():
        raise ProductRLVRTrainError("warm-start checkpoint is missing")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backbone, resolved_loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = ProductReasoningModel(
        backbone,
        args.arm,
        args.lora_layers,
        args.lora_rank,
        args.lora_alpha,
        workspace_width=512,
        workspace_slots=16,
        recurrent_steps=8,
        dense_width=192,
        unfreeze_layers=args.unfreeze_layers,
    ).to("cuda:0")
    warm_update, warm_metadata = load_trainable_checkpoint(
        args.warm_start_checkpoint, model
    )
    validate_warm_start_metadata(warm_metadata, args)
    warm_sha256 = _sha256_file(args.warm_start_checkpoint)

    reward_rows, reward_data_sha256 = _reservoir_reward_rows(
        args.data, args.max_rows, args.data_seed
    )
    _validate_reward_rows(reward_rows)
    replay_rows, replay_data_sha256 = reservoir_rows_with_sha256(
        args.replay_data, args.replay_max_rows, args.replay_data_seed
    )
    _validate_resume_contract(
        warm_update,
        warm_metadata,
        args,
        reward_data_sha256,
        replay_data_sha256,
    )
    if len(reward_rows) < args.groups_per_update:
        raise ProductRLVRTrainError("reward population is smaller than one update")
    stop_token_ids = _generation_stop_token_ids(tokenizer)

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    if args.start_update:
        _restore_optimizer(args.warm_start_checkpoint, optimizer)
    metadata = {
        "arm": args.arm,
        "model_root": str((args.model_source_root or args.model_root).resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_loader,
        "backbone_layout": model.backbone_layout,
        "data": str(args.data.resolve()),
        "data_sha256": reward_data_sha256,
        "selected_rows": len(reward_rows),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "lora_layers": args.lora_layers,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_projection_count": model.lora_projection_count,
        "unfreeze_layers": model.unfreeze_layers,
        "trainable_parameters": model.trainable_parameter_count(),
        "workspace_config": None,
        "workspace_architecture_sha256": None,
        "warm_start_checkpoint": str(args.warm_start_checkpoint.resolve()),
        "warm_start_sha256": warm_sha256,
        "warm_start_update": warm_update,
        "warm_start_data_sha256": warm_metadata.get("data_sha256"),
        "rlvr_algorithm": "single_use_on_policy_group_normalized_reinforce_v1",
        "rlvr_samples": args.samples,
        "rlvr_groups_per_update": args.groups_per_update,
        "rlvr_max_new_tokens": args.max_new_tokens,
        "rlvr_reward": "exact_math_answer_with_explicit_marker_and_terminal_stop_v1",
        "rlvr_replay_weight": args.replay_weight,
        "rlvr_replay_data": str(args.replay_data.resolve()),
        "rlvr_replay_data_sha256": replay_data_sha256,
        "rlvr_replay_rows": len(replay_rows),
        "rlvr_replay_data_seed": args.replay_data_seed,
        "rlvr_start_update": args.start_update,
        "rlvr_schedule_total_updates": args.schedule_total_updates,
    }

    model.train()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    update = args.start_update
    reward_cursor = args.start_update * args.groups_per_update
    replay_cursor = args.start_update * args.groups_per_update
    charged_tokens = 0
    generated_tokens = 0
    correct_candidates = 0
    rewarded_candidates = 0
    correct_exhausted_candidates = 0
    candidates = 0
    mixed_groups = 0
    uniform_groups = 0
    rollout_ledger: list[dict[str, Any]] = []
    trace: list[dict[str, float | int]] = []
    while update < args.updates:
        batch_rows = [
            reward_rows[(reward_cursor + offset) % len(reward_rows)]
            for offset in range(args.groups_per_update)
        ]
        reward_cursor += args.groups_per_update
        generation_seed = args.seed + update * 1_000_003
        groups = _generate_group_batch(
            model,
            tokenizer,
            batch_rows,
            samples=args.samples,
            max_new_tokens=args.max_new_tokens,
            seed=generation_seed,
            stop_token_ids=stop_token_ids,
        )

        update_reward = 0.0
        update_mixed = 0
        update_policy_logp = 0.0
        update_policy_terms = 0
        for row, group in zip(batch_rows, groups, strict=True):
            rewards = torch.tensor(
                [verified_terminal_reward(candidate) for candidate in group],
                device="cuda:0",
            )
            advantages = standardized_group_advantages(rewards)
            is_mixed = bool(torch.count_nonzero(advantages))
            mixed_groups += int(is_mixed)
            uniform_groups += int(not is_mixed)
            update_mixed += int(is_mixed)
            update_reward += float(rewards.mean())
            for candidate, advantage in zip(group, advantages, strict=True):
                generated_tokens += int(candidate["generated_tokens"])
                correct_candidates += int(candidate["correct"])
                rewarded_candidates += int(verified_terminal_reward(candidate))
                correct_exhausted_candidates += int(
                    candidate["correct"] and candidate["max_token_exhausted"]
                )
                candidates += 1
                rollout_ledger.append(
                    {
                        "schema": "shohin-product-rlvr-rollout-v1",
                        "update": update + 1,
                        "generation_seed": generation_seed,
                        "identity_sha256": row["identity_sha256"],
                        "question": row["question"],
                        "sample_index": candidate["sample_index"],
                        "completion": candidate["completion"],
                        "prediction": candidate["prediction"],
                        "gold": candidate["gold"],
                        "correct": candidate["correct"],
                        "reward": verified_terminal_reward(candidate),
                        "explicit_final_answer": candidate["explicit_final_answer"],
                        "generated_tokens": candidate["generated_tokens"],
                        "max_token_exhausted": candidate["max_token_exhausted"],
                        "advantage": float(advantage),
                    }
                )
                if not float(advantage):
                    continue
                if (
                    len(candidate["prompt_ids"]) + len(candidate["response_ids"])
                    > args.max_sequence_length
                ):
                    raise ProductRLVRTrainError(
                        "an on-policy trajectory exceeds the training context"
                    )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logp, tokens = _average_logp(
                        model,
                        candidate["prompt_ids"],
                        candidate["response_ids"],
                        tokenizer.pad_token_id,
                    )
                    objective = policy_objective(logp, advantage) / (
                        args.groups_per_update * args.samples
                    )
                objective.backward()
                charged_tokens += tokens
                update_policy_logp += float(logp.detach())
                update_policy_terms += 1

        replay_logp_sum = 0.0
        for _ in range(args.groups_per_update):
            replay_row = replay_rows[replay_cursor % len(replay_rows)]
            replay_cursor += 1
            replay_prompt, replay_response = _tokenize_rows(
                tokenizer,
                [replay_row],
                args.max_sequence_length,
                workspace_slots=0,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                replay_logp, replay_tokens = _average_logp(
                    model,
                    replay_prompt[0],
                    replay_response[0],
                    tokenizer.pad_token_id,
                )
                replay_objective = -(
                    args.replay_weight * replay_logp / args.groups_per_update
                )
            replay_objective.backward()
            charged_tokens += replay_tokens
            replay_logp_sum += float(replay_logp.detach())

        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        progress = update / max(args.schedule_total_updates - 1, 1)
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update == 1 or update % args.log_interval == 0:
            elapsed = time.monotonic() - started
            event: dict[str, float | int] = {
                "update": update,
                "reward_rate": update_reward / args.groups_per_update,
                "mixed_groups": update_mixed,
                "policy_mean_logp": (
                    update_policy_logp / update_policy_terms
                    if update_policy_terms
                    else 0.0
                ),
                "policy_terms": update_policy_terms,
                "replay_mean_logp": replay_logp_sum / args.groups_per_update,
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "generated_tokens": generated_tokens,
                "charged_tokens": charged_tokens,
                "elapsed_seconds": elapsed,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
        if update % args.checkpoint_interval == 0 or update == args.updates:
            _save_checkpoint(
                args.output / f"checkpoint_{update:07d}.pt",
                model,
                optimizer,
                update,
                metadata,
            )

    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    ledger_path = args.output / "rollouts.jsonl"
    ledger_sha256 = _atomic_jsonl(ledger_path, rollout_ledger)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        **metadata,
        "updates": update,
        "chunk_updates": update - args.start_update,
        "learning_rate": args.learning_rate,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "charged_tokens": charged_tokens,
        "charged_tokens_per_second": charged_tokens / elapsed,
        "candidates": candidates,
        "correct_candidates": correct_candidates,
        "candidate_accuracy": correct_candidates / candidates,
        "rewarded_candidates": rewarded_candidates,
        "reward_rate": rewarded_candidates / candidates,
        "correct_exhausted_candidates": correct_exhausted_candidates,
        "mixed_groups": mixed_groups,
        "uniform_groups": uniform_groups,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "rollout_ledger": str(ledger_path.resolve()),
        "rollout_ledger_sha256": ledger_sha256,
        "trace": trace,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--model-loader", choices=("auto", "causal", "multimodal"), default="auto"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--replay-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline",), default="baseline")
    parser.add_argument("--start-update", type=int, default=0)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--schedule-total-updates", type=int, default=100)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--groups-per-update", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--replay-max-rows", type=int, default=100000)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=2e-7)
    parser.add_argument("--lora-layers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--unfreeze-layers", type=int, default=2)
    parser.add_argument("--replay-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--data-seed", type=int, default=20260804)
    parser.add_argument("--replay-data-seed", type=int, default=20260802)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.schedule_total_updates,
        args.samples,
        args.groups_per_update,
        args.max_rows,
        args.replay_max_rows,
        args.max_sequence_length,
        args.max_new_tokens,
        args.learning_rate,
        args.lora_layers,
        args.lora_rank,
        args.lora_alpha,
        args.replay_weight,
        args.log_interval,
        args.checkpoint_interval,
    )
    if any(value <= 0 for value in positive) or args.unfreeze_layers < 0:
        parser.error("RLVR dimensions must be positive")
    if not 0 <= args.start_update < args.updates <= args.schedule_total_updates:
        parser.error("RLVR updates must satisfy 0 <= start < end <= schedule total")
    if not 2 <= args.samples <= 8:
        parser.error("RLVR samples must be in [2, 8]")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        f"[product-rlvr] updates={report['updates']} "
        f"answer_accuracy={report['candidate_accuracy']:.4f} "
        f"reward={report['reward_rate']:.4f} "
        f"generated_tokens/s={report['generated_tokens_per_second']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
