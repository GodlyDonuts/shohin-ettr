#!/usr/bin/env python3
"""Matched PCSD falsifier on depth-shifted conservative register programs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from prompt_conditioned_syndrome import (
    MinimumNormSyndromeProjector,
    PromptConditionedCheckCompiler,
    StickyChecks,
    SyndromeConfig,
    TiedStateProposer,
)


SCHEMA = "shohin-pcsd-conservation-shift-v1"
Arm = Literal["pcsd", "dense"]
EvaluationSplit = Literal["development", "confirmation"]

EVALUATION_SPECS: dict[EvaluationSplit, tuple[tuple[int, int], ...]] = {
    "development": ((8, 41008), (12, 41012)),
    "confirmation": ((16, 91016), (32, 91032)),
}


class ConservationShiftError(RuntimeError):
    """The matched architecture gate violated its frozen contract."""


@dataclass(frozen=True, slots=True)
class LedgerConfig:
    registers: int = 8
    modulus: int = 11
    width: int = 64
    checks: int = 4
    heads: int = 4
    ff_multiplier: int = 2
    maximum_depth: int = 32

    def validate(self) -> None:
        if min(
            self.registers,
            self.modulus,
            self.width,
            self.checks,
            self.heads,
            self.ff_multiplier,
            self.maximum_depth,
        ) <= 0:
            raise ConservationShiftError("ledger dimensions must be positive")
        if self.width % self.heads:
            raise ConservationShiftError("ledger width must divide across heads")
        if self.checks > min(self.registers, self.width):
            raise ConservationShiftError("ledger checks exceed factorized rank")


@dataclass(frozen=True, slots=True)
class LedgerBatch:
    initial: torch.Tensor
    operations: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    query: torch.Tensor
    answer: torch.Tensor


def _scatter(values: torch.Tensor, index: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    return values.scatter(1, index.unsqueeze(1), update.unsqueeze(1))


def execute_program(
    initial: torch.Tensor,
    operations: torch.Tensor,
    modulus: int,
) -> torch.Tensor:
    """Execute TRANSFER, SWAP, and ROTATE3 programs exactly."""

    if initial.ndim != 2 or operations.ndim != 3 or operations.shape[-1] != 5:
        raise ConservationShiftError("ledger program geometry differs")
    state = initial.clone()
    for step in range(operations.shape[1]):
        opcode, first, second, third, amount = operations[:, step].unbind(-1)
        first_value = state.gather(1, first.unsqueeze(1)).squeeze(1)
        second_value = state.gather(1, second.unsqueeze(1)).squeeze(1)
        third_value = state.gather(1, third.unsqueeze(1)).squeeze(1)

        transfer = opcode.eq(0)
        transferred_first = (first_value - amount) % modulus
        transferred_second = (second_value + amount) % modulus
        next_state = _scatter(state, first, torch.where(transfer, transferred_first, first_value))
        next_state = _scatter(
            next_state,
            second,
            torch.where(transfer, transferred_second, second_value),
        )

        swap = opcode.eq(1)
        next_state = _scatter(
            next_state,
            first,
            torch.where(swap, second_value, next_state.gather(1, first[:, None]).squeeze(1)),
        )
        next_state = _scatter(
            next_state,
            second,
            torch.where(swap, first_value, next_state.gather(1, second[:, None]).squeeze(1)),
        )

        rotate = opcode.eq(2)
        next_state = _scatter(
            next_state,
            first,
            torch.where(rotate, second_value, next_state.gather(1, first[:, None]).squeeze(1)),
        )
        next_state = _scatter(
            next_state,
            second,
            torch.where(rotate, third_value, next_state.gather(1, second[:, None]).squeeze(1)),
        )
        next_state = _scatter(
            next_state,
            third,
            torch.where(rotate, first_value, next_state.gather(1, third[:, None]).squeeze(1)),
        )
        state = next_state
    return state


def generate_batch(
    batch_size: int,
    depth: int,
    config: LedgerConfig,
    *,
    seed: int,
    device: torch.device,
) -> LedgerBatch:
    config.validate()
    if batch_size <= 0 or not 1 <= depth <= config.maximum_depth:
        raise ConservationShiftError("batch size or depth differs")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randint(
        config.modulus, (batch_size, config.registers), generator=generator
    )
    opcode = torch.randint(3, (batch_size, depth), generator=generator)
    first = torch.randint(config.registers, (batch_size, depth), generator=generator)
    offset = torch.randint(
        1, config.registers, (batch_size, depth), generator=generator
    )
    second = (first + offset) % config.registers
    third_offset = torch.randint(
        1, config.registers, (batch_size, depth), generator=generator
    )
    third = (second + third_offset) % config.registers
    third = torch.where(third.eq(first), (third + 1) % config.registers, third)
    amount = torch.randint(config.modulus, (batch_size, depth), generator=generator)
    operations = torch.stack((opcode, first, second, third, amount), dim=-1)
    target = execute_program(initial, operations, config.modulus)
    query = torch.randint(config.registers, (batch_size,), generator=generator)
    answer = target.gather(1, query.unsqueeze(1)).squeeze(1)
    source = torch.cat((initial, operations.flatten(1)), dim=1)
    return LedgerBatch(
        initial=initial.to(device),
        operations=operations.to(device),
        source=source.to(device),
        target=target.to(device),
        query=query.to(device),
        answer=answer.to(device),
    )


def batch_sha256(batch: LedgerBatch) -> str:
    digest = hashlib.sha256()
    for tensor in (
        batch.initial,
        batch.operations,
        batch.target,
        batch.query,
        batch.answer,
    ):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FactorizedDenseCorrector(nn.Module):
    """A similarly sized learned residual with no explicit constraint solve."""

    def __init__(self, config: SyndromeConfig):
        super().__init__()
        self.basis = nn.Parameter(
            torch.empty(config.checks, config.slots, config.state_width)
        )
        self.coefficients = nn.Linear(config.state_width, config.checks)
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.basis, std=0.02)

    def forward(self, proposed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = torch.tanh(self.coefficients(proposed.mean(1)))
        correction = torch.einsum("bc,csw->bsw", coefficients, self.basis)
        corrected = proposed + self.scale.tanh() * correction
        return corrected, correction


class ConservationReasoner(nn.Module):
    def __init__(self, config: LedgerConfig, arm: Arm):
        super().__init__()
        config.validate()
        if arm not in ("pcsd", "dense"):
            raise ConservationShiftError("unknown architecture arm")
        self.config = config
        self.arm = arm
        syndrome_config = SyndromeConfig(
            input_width=config.width,
            state_width=config.width,
            slots=config.registers,
            checks=config.checks,
            heads=config.heads,
            steps=config.maximum_depth,
            min_steps=0,
            ff_multiplier=config.ff_multiplier,
            use_step_embedding=False,
        )
        self.syndrome_config = syndrome_config
        self.value_embedding = nn.Embedding(config.modulus, config.width)
        self.register_embedding = nn.Embedding(config.registers, config.width)
        self.operation_embedding = nn.Embedding(3, config.width)
        self.amount_embedding = nn.Embedding(config.modulus, config.width)
        self.command_norm = nn.RMSNorm(config.width)
        self.compiler = (
            PromptConditionedCheckCompiler(syndrome_config)
            if arm == "pcsd"
            else None
        )
        self.projector = (
            MinimumNormSyndromeProjector(syndrome_config)
            if arm == "pcsd"
            else None
        )
        self.dense_corrector = (
            FactorizedDenseCorrector(syndrome_config) if arm == "dense" else None
        )
        self.proposer = TiedStateProposer(syndrome_config)
        self.state_norm = nn.RMSNorm(config.width)
        self.state_head = nn.Linear(config.width, config.modulus)
        self.query_seed = nn.Parameter(torch.empty(config.width))
        self.query_attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.answer_head = nn.Linear(config.width, config.modulus)
        nn.init.normal_(self.query_seed, std=0.02)

    def _initial_state(self, initial: torch.Tensor) -> torch.Tensor:
        registers = torch.arange(
            self.config.registers, device=initial.device
        ).unsqueeze(0)
        return self.state_norm(
            self.value_embedding(initial) + self.register_embedding(registers)
        )

    def _commands(self, operations: torch.Tensor) -> torch.Tensor:
        opcode, first, second, third, amount = operations.unbind(-1)
        return self.command_norm(
            self.operation_embedding(opcode)
            + self.register_embedding(first)
            + self.register_embedding(second)
            + self.register_embedding(third)
            + self.amount_embedding(amount)
        )

    def forward(
        self,
        batch: LedgerBatch,
        *,
        perturb_step: int | None = None,
        perturb_scale: float = 0.0,
        perturb_seed: int = 0,
        disable_projection: bool = False,
        shuffled_checks: bool = False,
    ) -> dict[str, torch.Tensor]:
        state = self._initial_state(batch.initial)
        commands = self._commands(batch.operations)
        source = torch.cat((state, commands), dim=1)
        source_mask = torch.ones(source.shape[:2], device=source.device, dtype=torch.bool)
        checks: StickyChecks | None = None
        if self.arm == "pcsd":
            assert self.compiler is not None
            checks = self.compiler(source, source_mask, state)
            if shuffled_checks:
                permutation = torch.arange(state.shape[0], device=state.device).roll(1)
                checks = StickyChecks(
                    slot_factors=checks.slot_factors[permutation],
                    feature_factors=checks.feature_factors,
                    reference_syndrome=checks.reference_syndrome[permutation],
                )

        pre_norms = []
        post_norms = []
        correction_norms = []
        one_mask = torch.ones(state.shape[0], 1, device=state.device, dtype=torch.bool)
        for step in range(commands.shape[1]):
            update, gate, _ = self.proposer(
                state, commands[:, step : step + 1], one_mask, step
            )
            proposed = state + gate * update
            if perturb_step == step and perturb_scale > 0:
                generator = torch.Generator(device=state.device).manual_seed(perturb_seed)
                proposed = proposed + perturb_scale * torch.randn(
                    proposed.shape,
                    generator=generator,
                    device=proposed.device,
                    dtype=proposed.dtype,
                )
            if self.arm == "pcsd" and not disable_projection:
                assert self.projector is not None and checks is not None
                state, pre, post, correction = self.projector(proposed, checks)
                pre_norms.append(pre.square().mean(-1).sqrt())
                post_norms.append(post.square().mean(-1).sqrt())
            elif self.arm == "dense":
                assert self.dense_corrector is not None
                state, correction = self.dense_corrector(proposed)
            else:
                state = proposed
                correction = torch.zeros_like(state)
            correction_norms.append(correction.square().mean((-2, -1)).sqrt())

        state_logits = self.state_head(self.state_norm(state))
        query = (
            self.register_embedding(batch.query) + self.query_seed.unsqueeze(0)
        ).unsqueeze(1)
        read, _ = self.query_attention(query, state, state, need_weights=False)
        answer_logits = self.answer_head(read.squeeze(1))
        zeros = torch.zeros(
            state.shape[0], commands.shape[1], device=state.device, dtype=state.dtype
        )
        return {
            "state": state,
            "state_logits": state_logits,
            "answer_logits": answer_logits,
            "pre_syndrome": torch.stack(pre_norms, 1) if pre_norms else zeros,
            "post_syndrome": torch.stack(post_norms, 1) if post_norms else zeros,
            "correction_rms": torch.stack(correction_norms, 1),
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _metrics(output: dict[str, torch.Tensor], batch: LedgerBatch) -> dict[str, float]:
    state_prediction = output["state_logits"].argmax(-1)
    answer_prediction = output["answer_logits"].argmax(-1)
    state_exact = state_prediction.eq(batch.target).all(-1)
    answer_exact = answer_prediction.eq(batch.answer)
    invariant = (state_prediction.sum(-1) - batch.initial.sum(-1)).remainder(
        int(output["state_logits"].shape[-1])
    )
    return {
        "answer_accuracy": answer_exact.float().mean().item(),
        "state_exact_accuracy": state_exact.float().mean().item(),
        "invariant_accuracy": invariant.eq(0).float().mean().item(),
        "pre_syndrome_rms": output["pre_syndrome"].mean().item(),
        "post_syndrome_rms": output["post_syndrome"].mean().item(),
        "correction_rms": output["correction_rms"].mean().item(),
    }


@torch.inference_mode()
def evaluate(
    model: ConservationReasoner,
    *,
    depth: int,
    count: int,
    seed: int,
    device: torch.device,
    perturb: bool = False,
    disable_projection: bool = False,
    shuffled_checks: bool = False,
) -> dict[str, Any]:
    model.eval()
    batch = generate_batch(count, depth, model.config, seed=seed, device=device)
    output = model(
        batch,
        perturb_step=(depth // 2 if perturb else None),
        perturb_scale=(0.5 if perturb else 0.0),
        perturb_seed=seed + 17,
        disable_projection=disable_projection,
        shuffled_checks=shuffled_checks,
    )
    return {
        "depth": depth,
        "count": count,
        "seed": seed,
        "batch_sha256": batch_sha256(batch),
        "perturb": perturb,
        "disable_projection": disable_projection,
        "shuffled_checks": shuffled_checks,
        **_metrics(output, batch),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ConservationShiftError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def evaluation_specs(split: EvaluationSplit) -> tuple[tuple[int, int], ...]:
    """Return the frozen cohort coordinates visible in one evaluation phase."""

    try:
        return EVALUATION_SPECS[split]
    except KeyError as error:
        raise ConservationShiftError(f"unknown evaluation split: {split}") from error


def _load_checkpoint(
    path: Path,
    *,
    model: ConservationReasoner,
    arm: Arm,
    config: LedgerConfig,
    device: torch.device,
) -> dict[str, Any]:
    if not path.is_file():
        raise ConservationShiftError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ConservationShiftError("checkpoint schema differs")
    if payload.get("arm") != arm or payload.get("config") != asdict(config):
        raise ConservationShiftError("checkpoint architecture differs")
    model.load_state_dict(payload["model"], strict=True)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise ConservationShiftError("CUDA is required unless --allow-cpu is explicit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = LedgerConfig(
        registers=args.registers,
        modulus=args.modulus,
        width=args.width,
        checks=args.checks,
        heads=args.heads,
        ff_multiplier=args.ff_multiplier,
        maximum_depth=args.maximum_depth,
    )
    cohort_manifest = None
    cohort_manifest_sha256 = None
    cohort_hashes: dict[tuple[int, int, int], tuple[str, str]] = {}
    if args.cohort_manifest is not None:
        cohort_bytes = args.cohort_manifest.read_bytes()
        cohort_manifest_sha256 = hashlib.sha256(cohort_bytes).hexdigest()
        cohort_manifest = json.loads(cohort_bytes)
        if cohort_manifest.get("schema") != "shohin-pcsd-conservation-shift-cohort-v1":
            raise ConservationShiftError("cohort manifest schema differs")
        if cohort_manifest.get("status") != "frozen":
            raise ConservationShiftError("cohort manifest is not frozen")
        if cohort_manifest.get("config") != asdict(config):
            raise ConservationShiftError("cohort manifest config differs")
        cohort_hashes = {
            (int(row["depth"]), int(row["seed"]), int(row["count"])): (
                str(row["sha256"]),
                str(row["split"]),
            )
            for row in cohort_manifest.get("batches") or []
        }
    model = ConservationReasoner(config, args.arm).to(device)
    started = time.monotonic()
    charged_examples = 0
    train_log = []
    source_checkpoint = None
    if args.eval_only:
        checkpoint_payload = _load_checkpoint(
            args.checkpoint,
            model=model,
            arm=args.arm,
            config=config,
            device=device,
        )
        source_checkpoint = args.checkpoint.resolve()
        checkpoint_updates = int(checkpoint_payload["updates"])
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        model.train()
        for update in range(1, args.updates + 1):
            depth = 2 + ((update * 997 + args.seed) % (args.train_depth_max - 1))
            batch = generate_batch(
                args.batch_size,
                depth,
                config,
                seed=args.data_seed + update,
                device=device,
            )
            perturb = update % 10 == 0
            output = model(
                batch,
                perturb_step=(depth // 2 if perturb else None),
                perturb_scale=(0.25 if perturb else 0.0),
                perturb_seed=args.data_seed + 100000 + update,
            )
            state_loss = F.cross_entropy(
                output["state_logits"].flatten(0, 1), batch.target.flatten()
            )
            answer_loss = F.cross_entropy(output["answer_logits"], batch.answer)
            loss = state_loss + answer_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            charged_examples += args.batch_size
            if update == 1 or update % args.log_every == 0 or update == args.updates:
                metrics = _metrics(output, batch)
                row = {
                    "update": update,
                    "depth": depth,
                    "loss": loss.item(),
                    "state_loss": state_loss.item(),
                    "answer_loss": answer_loss.item(),
                    **metrics,
                }
                train_log.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
        checkpoint_updates = args.updates

    elapsed = time.monotonic() - started
    evaluations = []
    selected_specs = evaluation_specs(args.evaluation_split)
    for depth, seed in selected_specs:
        evaluations.append(
            evaluate(
                model,
                depth=depth,
                count=args.eval_count,
                seed=seed,
                device=device,
            )
        )
        evaluations.append(
            evaluate(
                model,
                depth=depth,
                count=args.eval_count,
                seed=seed,
                device=device,
                perturb=True,
            )
        )
    if args.arm == "pcsd":
        for ablation in ("zero", "shuffled"):
            depth, seed = selected_specs[-1]
            evaluations.append(
                evaluate(
                    model,
                    depth=depth,
                    count=args.eval_count,
                    seed=seed,
                    device=device,
                    disable_projection=ablation == "zero",
                    shuffled_checks=ablation == "shuffled",
                )
            )

    if cohort_manifest is not None:
        expected_keys = {
            key
            for key, (_, split) in cohort_hashes.items()
            if split == args.evaluation_split
        }
        observed = {
            (int(row["depth"]), int(row["seed"]), int(row["count"])): str(
                row["batch_sha256"]
            )
            for row in evaluations
            if not row["perturb"]
        }
        if not expected_keys:
            raise ConservationShiftError("cohort manifest lacks evaluation split")
        if set(observed) != expected_keys or any(
            observed[key] != cohort_hashes[key][0] for key in expected_keys
        ):
            raise ConservationShiftError("generated evaluation cohort hash differs")

    if args.eval_only:
        checkpoint = args.checkpoint
    else:
        checkpoint = args.output.with_suffix(".pt")
        if checkpoint.exists():
            raise ConservationShiftError(f"refusing existing checkpoint: {checkpoint}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": SCHEMA,
                "arm": args.arm,
                "config": asdict(config),
                "model": model.state_dict(),
                "seed": args.seed,
                "data_seed": args.data_seed,
                "updates": args.updates,
            },
            checkpoint,
        )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "schema": SCHEMA,
        "status": "complete",
        "mode": "evaluation" if args.eval_only else "training",
        "arm": args.arm,
        "config": asdict(config),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "updates": checkpoint_updates,
        "batch_size": args.batch_size,
        "train_depth_max": args.train_depth_max,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "parameters": parameter_count(model),
        "charged_examples": charged_examples,
        "elapsed_seconds": elapsed,
        "examples_per_second": charged_examples / elapsed,
        "train_log": train_log,
        "evaluations": evaluations,
        "evaluation_split": args.evaluation_split,
        "checkpoint": str(checkpoint.resolve()),
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256,
        "cohort_manifest": (
            str(args.cohort_manifest.resolve()) if args.cohort_manifest else None
        ),
        "cohort_manifest_sha256": cohort_manifest_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("pcsd", "dense"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--data-seed", type=int, default=20260806)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-depth-max", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--registers", type=int, default=8)
    parser.add_argument("--modulus", type=int, default=11)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--checks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-multiplier", type=int, default=2)
    parser.add_argument("--maximum-depth", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--cohort-manifest", type=Path)
    parser.add_argument(
        "--evaluation-split",
        choices=("development", "confirmation"),
        default="development",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.train_depth_max <= args.maximum_depth:
        parser.error("train depth must be between 2 and maximum depth")
    if min(args.updates, args.batch_size, args.eval_count, args.log_every) <= 0:
        parser.error("training counts must be positive")
    if args.eval_only != (args.checkpoint is not None):
        parser.error("--eval-only and --checkpoint must be provided together")
    if args.evaluation_split == "confirmation" and not args.eval_only:
        parser.error("confirmation requires checkpoint-only evaluation")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    _atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"train_log", "evaluations"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
