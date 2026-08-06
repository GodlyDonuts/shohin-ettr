#!/usr/bin/env python3
"""Train and gate the frozen DIVERGE-QTG1 source interface."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from assess_diverge_hsc1_support_rank import load_frozen_hsc1
from diverge_mqb1_data import FIELD_COUNT, generate_mention_evidence
from diverge_qtg1_data import FIELD_QUERIES
from diverge_qtg1_runtime import (
    NONE_VALUE,
    REGISTER_COUNT,
    QTG1Config,
    QueryConditionedGatherer,
    QueryGatherLogits,
    architecture_receipt,
)
from diverge_sc1_neural_compiler import encode_source
from train_diverge_mqb1 import (
    COHORT_OFFSETS,
    EVIDENCE_COHORTS,
    _atomic_json,
    _atomic_torch,
    _load_qualified_mei1,
    _sha256,
    _state_sha256,
    renderer_parity,
)


SCHEMA = "shohin-diverge-qtg1-component-training-v1"
SEED = 202608058400
SOURCE_MODULES = ("memory_norm", "memory_projection", "memory_encoder")


def _source_adapter_state(source_model) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for module_name in SOURCE_MODULES:
        module = getattr(source_model.source, module_name)
        for name, value in module.state_dict().items():
            state[f"{module_name}.{name}"] = value.detach().cpu()
    return state


def _load_source_adapter_state(source_model, state: dict[str, torch.Tensor]) -> None:
    for module_name in SOURCE_MODULES:
        prefix = f"{module_name}."
        local = {
            name.removeprefix(prefix): value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        getattr(source_model.source, module_name).load_state_dict(local, strict=True)


def _enable_source_adapter(source_model) -> list[torch.nn.Parameter]:
    source_model.requires_grad_(False)
    parameters: list[torch.nn.Parameter] = []
    for module_name in SOURCE_MODULES:
        module = getattr(source_model.source, module_name)
        module.requires_grad_(True)
        parameters.extend(module.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("QTG1 source adapter did not become trainable")
    return parameters


def _conditioned_features(source_model, examples, device: torch.device):
    encodings = []
    query_lengths: list[int] = []
    evidence_starts: list[int] = []
    evidence_lengths: list[int] = []
    for row in examples:
        for query in FIELD_QUERIES:
            tokens = (*query, "evidence", *row.words)
            encodings.append(encode_source(source_model.source.tokenizer, tokens))
            query_lengths.append(len(query))
            evidence_starts.append(len(query) + 1)
            evidence_lengths.append(len(row.words))
    words, _ = source_model.source._encode_words(encodings, device)
    query_rows = [
        words[index, : query_lengths[index]] for index in range(len(encodings))
    ]
    evidence_rows = [
        words[
            index,
            evidence_starts[index] : evidence_starts[index] + evidence_lengths[index],
        ]
        for index in range(len(encodings))
    ]
    query = pad_sequence(query_rows, batch_first=True)
    evidence = pad_sequence(evidence_rows, batch_first=True)
    query_length = torch.tensor(query_lengths, device=device)
    evidence_length = torch.tensor(evidence_lengths, device=device)
    query_mask = torch.arange(query.shape[1], device=device)[None] < query_length[:, None]
    evidence_mask = (
        torch.arange(evidence.shape[1], device=device)[None] < evidence_length[:, None]
    )
    batch = len(examples)
    query = query.reshape(batch, FIELD_COUNT, query.shape[1], query.shape[2])
    evidence = evidence.reshape(
        batch, FIELD_COUNT, evidence.shape[1], evidence.shape[2]
    )
    query_mask = query_mask.reshape(batch, FIELD_COUNT, query_mask.shape[1])
    evidence_mask = evidence_mask.reshape(batch, FIELD_COUNT, evidence_mask.shape[1])
    pointer = torch.full((batch, FIELD_COUNT), -1, dtype=torch.long, device=device)
    states = torch.tensor(
        [(*row.before, *row.after) for row in examples],
        dtype=torch.long,
        device=device,
    )
    for row_index, row in enumerate(examples):
        for mention in row.mentions:
            pointer[row_index, mention.field] = mention.word_index
    if pointer.lt(0).any():
        raise ValueError("QTG1 supervisor lost a typed mention")
    value_target = torch.full(
        evidence_mask.shape, NONE_VALUE, dtype=torch.long, device=device
    )
    rows = torch.arange(batch, device=device)[:, None]
    fields = torch.arange(FIELD_COUNT, device=device)[None]
    value_target[rows, fields, pointer] = states
    return query, query_mask, evidence, evidence_mask, pointer, value_target, states


def _balanced_value_loss(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
    ).reshape_as(target)
    positive = mask & target.ne(NONE_VALUE)
    negative = mask & target.eq(NONE_VALUE)
    return 0.5 * (loss[positive].mean() + loss[negative].mean())


def _train_batch(
    model: QueryConditionedGatherer,
    source_model,
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    examples = [
        generate_mention_evidence(seed=seed * batch_size + index, cohort="train")
        for index in range(batch_size)
    ]
    query, query_mask, evidence, evidence_mask, pointer, value_target, states = (
        _conditioned_features(source_model, examples, device)
    )
    logits = model(query, query_mask, evidence, evidence_mask)
    pointer_loss = F.cross_entropy(
        logits.pointer.reshape(-1, logits.pointer.shape[-1]), pointer.reshape(-1)
    )
    value_loss = _balanced_value_loss(logits.value, value_target, evidence_mask)
    loss = pointer_loss + value_loss
    with torch.no_grad():
        rows = torch.arange(batch_size, device=device)[:, None]
        fields = torch.arange(FIELD_COUNT, device=device)[None]
        pointer_prediction = logits.pointer.argmax(-1)
        value_prediction = logits.value.argmax(-1)[rows, fields, pointer]
    return loss, {
        "loss": float(loss.detach()),
        "pointer_loss": float(pointer_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "pointer_exact": float(pointer_prediction.eq(pointer).float().mean()),
        "gold_value_exact": float(value_prediction.eq(states).float().mean()),
    }


def _reverse_valid(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    permutation = torch.arange(mask.shape[-1], device=mask.device)[None, None].expand(
        mask.shape[0], mask.shape[1], -1
    ).clone()
    for row in range(mask.shape[0]):
        valid = torch.nonzero(mask[row, 0], as_tuple=False).squeeze(-1)
        permutation[row, :, valid] = valid.flip(0)[None]
    expansion = permutation
    while expansion.ndim < tensor.ndim:
        expansion = expansion.unsqueeze(-1)
    return tensor.gather(2, expansion.expand_as(tensor))


@torch.no_grad()
def evaluate_cohort(
    model: QueryConditionedGatherer,
    source_model,
    *,
    cohort: str,
    count: int,
    seed: int,
    batch_size: int,
    device: torch.device,
    controls: bool,
) -> dict[str, float | int]:
    totals = {
        "valid": 0,
        "before": 0,
        "after": 0,
        "complete": 0,
        "assignment": 0,
        "value_fields": 0,
        "provenance_mismatches": 0,
        "accepted_duplicates": 0,
        "accepted_overflow": 0,
        "query_shuffle_complete": 0,
        "value_shuffle_complete": 0,
    }
    for start in range(0, count, batch_size):
        examples = [
            generate_mention_evidence(seed=seed + index, cohort=cohort)
            for index in range(start, min(count, start + batch_size))
        ]
        query, query_mask, evidence, evidence_mask, pointer, _, states = (
            _conditioned_features(source_model, examples, device)
        )
        logits = model(query, query_mask, evidence, evidence_mask)
        binding = model.decode(logits, evidence_mask)
        before_ok = binding.before.eq(states[:, :REGISTER_COUNT]).all(-1)
        after_ok = binding.after.eq(states[:, REGISTER_COUNT:]).all(-1)
        totals["valid"] += int(binding.valid.sum())
        totals["before"] += int((binding.valid & before_ok).sum())
        totals["after"] += int((binding.valid & after_ok).sum())
        totals["complete"] += int((binding.valid & before_ok & after_ok).sum())
        totals["assignment"] += int(
            (binding.valid & binding.provenance.eq(pointer).all(-1)).sum()
        )
        totals["value_fields"] += int(binding.selected_values.eq(states).sum())
        totals["provenance_mismatches"] += int(
            (binding.valid[:, None] & binding.provenance.ne(pointer)).sum()
        )
        sorted_words = binding.provenance.sort(-1).values
        duplicate = sorted_words[:, 1:].eq(sorted_words[:, :-1]).any(-1)
        totals["accepted_duplicates"] += int((binding.valid & duplicate).sum())
        totals["accepted_overflow"] += int((binding.valid & binding.overflow).sum())
        if controls:
            query_shuffled = model.decode(
                replace(
                    logits,
                    pointer=logits.pointer.roll(1, dims=1),
                    value=logits.value.roll(1, dims=1),
                    field=logits.field.roll(1, dims=1),
                ),
                evidence_mask,
            )
            totals["query_shuffle_complete"] += int(
                (
                    query_shuffled.valid
                    & query_shuffled.before.eq(states[:, :REGISTER_COUNT]).all(-1)
                    & query_shuffled.after.eq(states[:, REGISTER_COUNT:]).all(-1)
                ).sum()
            )
            value_shuffled = model.decode(
                replace(logits, value=_reverse_valid(logits.value, evidence_mask)),
                evidence_mask,
            )
            totals["value_shuffle_complete"] += int(
                (
                    value_shuffled.valid
                    & value_shuffled.before.eq(states[:, :REGISTER_COUNT]).all(-1)
                    & value_shuffled.after.eq(states[:, REGISTER_COUNT:]).all(-1)
                ).sum()
            )
    result: dict[str, float | int] = {
        "examples": count,
        "valid_rate": totals["valid"] / count,
        "before_state_exact": totals["before"] / count,
        "after_state_exact": totals["after"] / count,
        "complete_state_pair_exact": totals["complete"] / count,
        "complete_assignment_exact": totals["assignment"] / count,
        "selected_value_exact": totals["value_fields"] / (count * FIELD_COUNT),
        "provenance_mismatches": totals["provenance_mismatches"],
        "accepted_duplicate_mentions": totals["accepted_duplicates"],
        "accepted_overflow": totals["accepted_overflow"],
    }
    if controls:
        result.update(
            {
                "query_shuffle_complete": totals["query_shuffle_complete"] / count,
                "value_shuffle_complete": totals["value_shuffle_complete"] / count,
            }
        )
    return result


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
        pair_width=args.source_pair_width,
        local_layers=args.local_layers,
        local_heads=args.local_heads,
    )
    source_parameters = _enable_source_adapter(source_model)
    source_initial = _source_adapter_state(source_model)
    source_initial_hash = _state_sha256(source_initial)
    backbone_initial_hash = _state_sha256(
        {name: value.detach().cpu() for name, value in source_model.source.backbone.state_dict().items()}
    )
    mei1, _, mei1_hash_before = _load_qualified_mei1(args.mei1_checkpoint, device)
    config = QTG1Config(
        input_width=args.input_width,
        width=args.width,
        heads=args.heads,
        layers=args.layers,
        pointer_width=args.pointer_width,
    )
    model = QueryConditionedGatherer(config).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "lr": args.gatherer_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": source_parameters,
                "lr": args.source_lr,
                "weight_decay": args.weight_decay,
            },
        ],
        betas=(0.9, 0.95),
    )
    parity = renderer_parity(args.parity_count)
    started = time.time()
    last: dict[str, float] = {}
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
        trainable = list(model.parameters()) + source_parameters
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            print(
                json.dumps(
                    {
                        "update": update,
                        "elapsed": time.time() - started,
                        "grad_norm": float(grad_norm),
                        **last,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    model.eval()
    evaluations = {
        cohort: evaluate_cohort(
            model,
            source_model,
            cohort=cohort,
            count=args.eval_count,
            seed=args.seed + 10_000_000 + COHORT_OFFSETS[cohort],
            batch_size=args.eval_batch_size,
            device=device,
            controls=True,
        )
        for cohort in EVIDENCE_COHORTS
    }
    source_adapted = _source_adapter_state(source_model)
    source_adapted_hash = _state_sha256(source_adapted)
    _load_source_adapter_state(source_model, source_initial)
    reset_evaluations = {
        cohort: evaluate_cohort(
            model,
            source_model,
            cohort=cohort,
            count=args.reset_eval_count,
            seed=args.seed + 50_000_000 + COHORT_OFFSETS[cohort],
            batch_size=args.eval_batch_size,
            device=device,
            controls=False,
        )
        for cohort in EVIDENCE_COHORTS
    }
    _load_source_adapter_state(source_model, source_adapted)
    backbone_final_hash = _state_sha256(
        {name: value.detach().cpu() for name, value in source_model.source.backbone.state_dict().items()}
    )
    mei1_hash_after = {
        "executor": _state_sha256(
            {
                name.removeprefix("executor."): value
                for name, value in mei1.state_dict().items()
                if name.startswith("executor.")
            }
        ),
        "query": _state_sha256(
            {
                name.removeprefix("query."): value
                for name, value in mei1.state_dict().items()
                if name.startswith("query.")
            }
        ),
    }
    shifted = [evaluations[name] for name in EVIDENCE_COHORTS if name != "train"]
    integrity = sum(
        int(row[metric])
        for row in evaluations.values()
        for metric in (
            "provenance_mismatches",
            "accepted_duplicate_mentions",
            "accepted_overflow",
        )
    )
    gate = {
        "renderer_parity_100pct": bool(parity["pass"]),
        "before_state_99pct_each": min(
            row["before_state_exact"] for row in evaluations.values()
        ) >= 0.99,
        "after_state_99pct_each": min(
            row["after_state_exact"] for row in evaluations.values()
        ) >= 0.99,
        "complete_assignment_99pct_each": min(
            row["complete_assignment_exact"] for row in evaluations.values()
        ) >= 0.99,
        "selected_value_99_9pct_each": min(
            row["selected_value_exact"] for row in evaluations.values()
        ) >= 0.999,
        "zero_integrity_failures": integrity == 0,
        "query_shuffle_drops_shifted_50pp_each": min(
            row["complete_state_pair_exact"] - row["query_shuffle_complete"]
            for row in shifted
        ) >= 0.50,
        "value_shuffle_drops_shifted_50pp_each": min(
            row["complete_state_pair_exact"] - row["value_shuffle_complete"]
            for row in shifted
        ) >= 0.50,
        "frozen_backbone_unchanged": backbone_initial_hash == backbone_final_hash,
        "frozen_algebra_query_unchanged": mei1_hash_before == mei1_hash_after,
        "candidate_source_audit": bool(architecture_receipt(model)["source_audit"]["pass"]),
    }
    gate["pass"] = all(gate.values())
    model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    elapsed = time.time() - started
    report = {
        "schema": SCHEMA,
        "status": "bounded-query-conditioned-source-gate-complete",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "architecture": {
            **architecture_receipt(model),
            "source_adapter_trainable_parameters": sum(
                parameter.numel() for parameter in source_parameters
            ),
            "total_trainable_parameters": model.trainable_parameters()
            + sum(parameter.numel() for parameter in source_parameters),
        },
        "inputs": {
            "base_sha256": _sha256(args.base),
            "tokenizer_sha256": _sha256(args.tokenizer),
            "sc1_sha256": _sha256(args.sc1_checkpoint),
            "hsc1_sha256": _sha256(args.hsc1_checkpoint),
            "mei1_sha256": _sha256(args.mei1_checkpoint),
        },
        "renderer_parity": parity,
        "training": {
            "elapsed_seconds": elapsed,
            "source_records": args.updates * args.batch_size,
            "query_conditioned_sequences": args.updates * args.batch_size * FIELD_COUNT,
            "updates": args.updates,
            "final_batch": last,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated())
            if device.type == "cuda"
            else 0,
            "gatherer_state_sha256": _state_sha256(model_state),
            "source_initial_sha256": source_initial_hash,
            "source_adapted_sha256": source_adapted_hash,
        },
        "evaluations": evaluations,
        "source_reset_evaluations": reset_evaluations,
        "frozen_backbone_before": backbone_initial_hash,
        "frozen_backbone_after": backbone_final_hash,
        "frozen_component_hashes_before": mei1_hash_before,
        "frozen_component_hashes_after": mei1_hash_after,
        "gate": gate,
        "claim_boundary": (
            "Synthetic query-conditioned source-interface gate only. Full DIVERGE "
            "composition remains blocked unless every conjunctive gate passes."
        ),
    }
    checkpoint = {
        "schema": SCHEMA,
        "config": asdict(config),
        "gatherer_state_dict": model_state,
        "source_adapter_state_dict": source_adapted,
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
    parser.add_argument("--mei1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--updates", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-count", type=int, default=20000)
    parser.add_argument("--reset-eval-count", type=int, default=2000)
    parser.add_argument("--parity-count", type=int, default=10000)
    parser.add_argument("--gatherer-lr", type=float, default=3e-4)
    parser.add_argument("--source-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--input-width", type=int, default=192)
    parser.add_argument("--source-pair-width", type=int, default=64)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--pointer-width", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite QTG1 output")
    if args.seed != SEED or args.updates != 1600 or args.batch_size != 64:
        raise ValueError("QTG1 training contract differs")
    if (
        args.eval_count != 20000
        or args.reset_eval_count != 2000
        or args.parity_count != 10000
    ):
        raise ValueError("QTG1 evaluation contract differs")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    report = run(args)
    if not report["gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
