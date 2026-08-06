#!/usr/bin/env python3
"""Train and component-gate the bounded DIVERGE-MEI1 neural interfaces."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Sequence

import torch
import torch.nn.functional as F

from assess_diverge_hsc1_support_rank import load_frozen_hsc1
from diverge_mei1_data import (
    EVIDENCE_COHORTS,
    exact_action_batch,
    exact_action_program,
    generate_probe_evidence,
    random_action_program,
    random_register_states,
)
from diverge_mei1_runtime import (
    DIVERGEMEI1,
    MEI1Config,
    REGISTER_COUNT,
    VALUE_COUNT,
    architecture_receipt,
)
from diverge_sc1_neural_compiler import encode_source


SCHEMA = "shohin-diverge-mei1-component-training-v1"
COHORT_OFFSETS = {
    "train": 0,
    "lexical_shift": 100_000,
    "renderer_shift": 200_000,
    "composition_shift": 300_000,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        raw = tensor.detach().cpu().contiguous()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode("utf-8"))
        digest.update(str(raw.dtype).encode("ascii"))
        digest.update(str(tuple(raw.shape)).encode("ascii"))
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _evidence_features(source_model, examples, device: torch.device):
    encodings = [encode_source(source_model.source.tokenizer, row.words) for row in examples]
    with torch.no_grad():
        words, lengths = source_model.source._encode_words(encodings, device)
    mask = torch.arange(words.shape[1], device=device)[None, :] < lengths[:, None]
    target = torch.tensor(
        [(*row.before, *row.after) for row in examples],
        dtype=torch.long,
        device=device,
    )
    return words, mask, target


def _probability_nll(probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    selected = probabilities.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return -selected.clamp_min(1e-12).log().mean()


def _train_batch(
    model: DIVERGEMEI1,
    source_model,
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    evidence = [
        generate_probe_evidence(seed=seed * batch_size + index, cohort="train")
        for index in range(batch_size)
    ]
    words, mask, evidence_target = _evidence_features(source_model, evidence, device)
    evidence_logits = model.evidence(words, mask)
    evidence_loss = F.cross_entropy(
        evidence_logits.reshape(-1, VALUE_COUNT), evidence_target.reshape(-1)
    )

    state_rows = random_register_states(seed + 20_000_000, batch_size)
    rng = random.Random(seed + 30_000_000)
    action_rows = tuple(rng.randrange(4) for _ in range(batch_size))
    successor_rows = exact_action_batch(state_rows, action_rows)
    states = torch.tensor(state_rows, dtype=torch.long, device=device)
    actions = torch.tensor(action_rows, dtype=torch.long, device=device)
    successor = torch.tensor(successor_rows, dtype=torch.long, device=device)
    probabilities, invalid_mass = model.executor(states, actions)
    executor_loss = _probability_nll(probabilities, successor)
    range_loss = invalid_mass.mean()

    query_states = random_register_states(seed + 40_000_000, batch_size)
    query_rng = random.Random(seed + 50_000_000)
    query_slots = tuple(query_rng.randrange(REGISTER_COUNT) for _ in range(batch_size))
    query_values = torch.tensor(query_states, dtype=torch.long, device=device)
    slots = torch.tensor(query_slots, dtype=torch.long, device=device)
    query_target = query_values.gather(1, slots.unsqueeze(-1)).squeeze(-1)
    query_probabilities = model.query(query_values, slots)
    query_loss = _probability_nll(query_probabilities, query_target)

    loss = evidence_loss + executor_loss + query_loss + 10.0 * range_loss
    with torch.no_grad():
        evidence_exact = evidence_logits.argmax(-1).eq(evidence_target).all(-1).float().mean()
        executor_exact = probabilities.argmax(-1).eq(successor).all(-1).float().mean()
        query_exact = query_probabilities.argmax(-1).eq(query_target).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "evidence_loss": float(evidence_loss.detach()),
        "executor_loss": float(executor_loss.detach()),
        "query_loss": float(query_loss.detach()),
        "invalid_mass": float(range_loss.detach()),
        "evidence_complete_exact": float(evidence_exact),
        "executor_complete_exact": float(executor_exact),
        "query_exact": float(query_exact),
    }


@torch.no_grad()
def evaluate_evidence(
    model: DIVERGEMEI1,
    source_model,
    *,
    cohort: str,
    count: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    before_exact = 0
    after_exact = 0
    field_exact = 0
    for start in range(0, count, batch_size):
        examples = [
            generate_probe_evidence(seed=seed + index, cohort=cohort)
            for index in range(start, min(count, start + batch_size))
        ]
        words, mask, target = _evidence_features(source_model, examples, device)
        predicted = model.evidence(words, mask).argmax(-1)
        before_exact += int(
            predicted[:, :REGISTER_COUNT]
            .eq(target[:, :REGISTER_COUNT])
            .all(-1)
            .sum()
        )
        after_exact += int(
            predicted[:, REGISTER_COUNT:]
            .eq(target[:, REGISTER_COUNT:])
            .all(-1)
            .sum()
        )
        field_exact += int(predicted.eq(target).sum())
    return {
        "examples": count,
        "before_state_exact": before_exact / count,
        "after_state_exact": after_exact / count,
        "field_exact": field_exact / (count * REGISTER_COUNT * 2),
    }


@torch.no_grad()
def evaluate_executor(
    model: DIVERGEMEI1,
    *,
    count: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    states = random_register_states(seed, count)
    rng = random.Random(seed + 1)
    actions = tuple(rng.randrange(4) for _ in range(count))
    target = exact_action_batch(states, actions)
    states_tensor = torch.tensor(states, dtype=torch.long, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target, dtype=torch.long, device=device)
    probabilities, invalid = model.executor(states_tensor, actions_tensor)
    predicted = model.executor.hard_step(states_tensor, actions_tensor)
    return {
        "examples": count,
        "complete_state_exact": float(predicted.eq(target_tensor).all(-1).float().mean()),
        "register_exact": float(predicted.eq(target_tensor).float().mean()),
        "maximum_invalid_mass": float(invalid.max()),
        "soft_target_probability": float(
            probabilities.gather(-1, target_tensor.unsqueeze(-1)).mean()
        ),
    }


@torch.no_grad()
def evaluate_depths(
    model: DIVERGEMEI1,
    *,
    count: int,
    seed: int,
    depths: Sequence[int],
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    output = {}
    for depth in depths:
        states = []
        programs = []
        targets = []
        rng = random.Random(seed + depth * 100_000)
        for index in range(count):
            values = tuple(rng.randrange(5, 45) for _ in range(REGISTER_COUNT))
            program = random_action_program(seed + depth * 1_000_000 + index, depth)
            states.append(values)
            programs.append(program)
            targets.append(exact_action_program(values, program))
        predicted = torch.tensor(states, dtype=torch.long, device=device)
        for step in range(depth):
            actions = torch.tensor(
                [program[step] for program in programs],
                dtype=torch.long,
                device=device,
            )
            predicted = model.executor.hard_step(predicted, actions)
        target = torch.tensor(targets, dtype=torch.long, device=device)
        output[str(depth)] = {
            "examples": count,
            "terminal_state_exact": float(predicted.eq(target).all(-1).float().mean()),
            "register_exact": float(predicted.eq(target).float().mean()),
        }
    return output


@torch.no_grad()
def evaluate_query(
    model: DIVERGEMEI1,
    *,
    count: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    states = random_register_states(seed, count)
    rng = random.Random(seed + 1)
    slots = tuple(rng.randrange(REGISTER_COUNT) for _ in range(count))
    state_tensor = torch.tensor(states, dtype=torch.long, device=device)
    slot_tensor = torch.tensor(slots, dtype=torch.long, device=device)
    target = state_tensor.gather(1, slot_tensor.unsqueeze(-1)).squeeze(-1)
    predicted = model.query.hard_read(state_tensor, slot_tensor)
    return {"examples": count, "exact": float(predicted.eq(target).float().mean())}


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    source_model = load_frozen_hsc1(
        base=args.base,
        tokenizer_path=args.tokenizer,
        sc1_checkpoint=args.sc1_checkpoint,
        hsc1_checkpoint=args.hsc1_checkpoint,
        device=device,
        layer=args.layer,
        width=args.input_width,
        pair_width=args.pair_width,
        local_layers=args.local_layers,
        local_heads=args.local_heads,
    )
    config = MEI1Config(
        input_width=args.input_width,
        evidence_width=args.evidence_width,
        evidence_heads=args.evidence_heads,
        evidence_layers=args.evidence_layers,
    )
    model = DIVERGEMEI1(config).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.evidence.parameters(),
                "lr": args.evidence_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": model.executor.parameters(),
                "lr": args.algebra_lr,
                "weight_decay": 0.0,
            },
            {
                "params": model.query.parameters(),
                "lr": args.algebra_lr,
                "weight_decay": 0.0,
            },
        ],
        betas=(0.9, 0.95),
    )
    started = time.time()
    last = {}
    model.train()
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, last = _train_batch(
            model,
            source_model,
            seed=args.seed + update,
            batch_size=args.batch_size,
            device=device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            print(
                json.dumps({"update": update, "elapsed": time.time() - started, **last}, sort_keys=True),
                flush=True,
            )
    model.eval()
    evidence = {
        cohort: evaluate_evidence(
            model,
            source_model,
            cohort=cohort,
            count=args.eval_evidence_count,
            seed=args.seed + 10_000_000 + COHORT_OFFSETS[cohort],
            batch_size=args.eval_batch_size,
            device=device,
        )
        for cohort in EVIDENCE_COHORTS
    }
    executor = evaluate_executor(
        model, count=args.eval_executor_count, seed=args.seed + 20_000_000, device=device
    )
    depths = evaluate_depths(
        model,
        count=args.eval_depth_count,
        seed=args.seed + 30_000_000,
        depths=args.eval_depths,
        device=device,
    )
    query = evaluate_query(
        model, count=args.eval_query_count, seed=args.seed + 40_000_000, device=device
    )
    gate = {
        "evidence_before_99pct_each": min(
            row["before_state_exact"] for row in evidence.values()
        ) >= 0.99,
        "evidence_after_99pct_each": min(
            row["after_state_exact"] for row in evidence.values()
        ) >= 0.99,
        "one_step_99_9pct": executor["complete_state_exact"] >= 0.999,
        "held_depth_99pct_each": min(
            row["terminal_state_exact"] for row in depths.values()
        ) >= 0.99,
        "query_99_9pct": query["exact"] >= 0.999,
        "zero_accepted_out_of_range_mass": executor["maximum_invalid_mass"] <= 1e-6,
        "candidate_source_audit": architecture_receipt(config)["candidate_source_audit"]["pass"],
    }
    gate["pass"] = all(gate.values())
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    report = {
        "schema": SCHEMA,
        "status": "bounded-component-training-complete",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "architecture": architecture_receipt(config),
        "inputs": {
            "base_sha256": _sha256(args.base),
            "tokenizer_sha256": _sha256(args.tokenizer),
            "sc1_sha256": _sha256(args.sc1_checkpoint),
            "hsc1_sha256": _sha256(args.hsc1_checkpoint),
        },
        "training": {
            "elapsed_seconds": time.time() - started,
            "examples": args.updates * args.batch_size,
            "final_batch": last,
            "model_state_sha256": _state_sha256(state),
        },
        "evidence": evidence,
        "executor": executor,
        "held_depths": depths,
        "query": query,
        "gate": gate,
        "claim_boundary": (
            "Component-island result only. Passing interfaces must still clear the frozen "
            "source-sealed full-composition gate."
        ),
    }
    checkpoint = {
        "schema": SCHEMA,
        "config": asdict(config),
        "state_dict": state,
        "model_state_sha256": report["training"]["model_state_sha256"],
        "report": report,
    }
    _atomic_torch(args.output, checkpoint)
    _atomic_json(args.report, report)
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "checkpoint_sha256": _sha256(args.output),
                "report": str(args.report),
                "report_sha256": _sha256(args.report),
                "gate": gate,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sc1-checkpoint", type=Path, required=True)
    parser.add_argument("--hsc1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202608057800)
    parser.add_argument("--updates", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-evidence-count", type=int, default=1024)
    parser.add_argument("--eval-executor-count", type=int, default=20000)
    parser.add_argument("--eval-depth-count", type=int, default=2000)
    parser.add_argument("--eval-query-count", type=int, default=20000)
    parser.add_argument("--eval-depths", type=int, nargs="+", default=(4, 8, 16, 24))
    parser.add_argument("--evidence-lr", type=float, default=3e-4)
    parser.add_argument("--algebra-lr", type=float, default=5e-2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--input-width", type=int, default=192)
    parser.add_argument("--pair-width", type=int, default=64)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-heads", type=int, default=4)
    parser.add_argument("--evidence-width", type=int, default=192)
    parser.add_argument("--evidence-heads", type=int, default=4)
    parser.add_argument("--evidence-layers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite MEI1 output")
    if min(
        args.updates,
        args.batch_size,
        args.eval_evidence_count,
        args.eval_executor_count,
        args.eval_depth_count,
        args.eval_query_count,
    ) <= 0:
        raise ValueError("MEI1 counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    report = run(args)
    if not report["gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
