#!/usr/bin/env python3
"""Train and gate the frozen DIVERGE-MQB1 structural mention binder."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

import torch
import torch.nn.functional as F

from assess_diverge_hsc1_support_rank import load_frozen_hsc1
from diverge_mei1_data import EVIDENCE_COHORTS, generate_probe_evidence
from diverge_mei1_runtime import DIVERGEMEI1, MEI1Config
from diverge_mqb1_data import FIELD_COUNT, generate_mention_evidence
from diverge_mqb1_runtime import (
    NONE_ADDRESS,
    NONE_PHASE,
    NONE_VALUE,
    REGISTER_COUNT,
    VALUE_COUNT,
    MQB1Config,
    MentionBinderLogits,
    MentionEvidenceBinder,
    architecture_receipt,
)
from diverge_sc1_neural_compiler import encode_source


SCHEMA = "shohin-diverge-mqb1-component-training-v1"
SEED = 202608058100
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


def _load_qualified_mei1(
    path: Path, device: torch.device
) -> tuple[DIVERGEMEI1, dict[str, object], dict[str, str]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "shohin-diverge-mei1-component-training-v1":
        raise ValueError("unexpected MEI1 checkpoint schema")
    model = DIVERGEMEI1(MEI1Config(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    observed = _state_sha256(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )
    if observed != payload.get("model_state_sha256"):
        raise ValueError("MEI1 model state digest differs")
    report = payload.get("report")
    if not isinstance(report, dict):
        raise ValueError("MEI1 checkpoint lacks its component report")
    if (
        report["executor"]["complete_state_exact"] < 0.999
        or min(row["terminal_state_exact"] for row in report["held_depths"].values()) < 0.99
        or report["query"]["exact"] < 0.999
    ):
        raise ValueError("MEI1 algebra/query components are not qualified")
    model.eval().requires_grad_(False)
    hashes = {
        "executor": _state_sha256(
            {
                name.removeprefix("executor."): value
                for name, value in model.state_dict().items()
                if name.startswith("executor.")
            }
        ),
        "query": _state_sha256(
            {
                name.removeprefix("query."): value
                for name, value in model.state_dict().items()
                if name.startswith("query.")
            }
        ),
    }
    return model, report, hashes


def _features_and_targets(source_model, examples, device: torch.device):
    encodings = [encode_source(source_model.source.tokenizer, row.words) for row in examples]
    with torch.no_grad():
        words, lengths = source_model.source._encode_words(encodings, device)
    mask = torch.arange(words.shape[1], device=device)[None, :] < lengths[:, None]
    batch, width = mask.shape
    value = torch.full((batch, width), NONE_VALUE, dtype=torch.long, device=device)
    phase = torch.full((batch, width), NONE_PHASE, dtype=torch.long, device=device)
    address = torch.full((batch, width), NONE_ADDRESS, dtype=torch.long, device=device)
    pointer = torch.full((batch, FIELD_COUNT), -1, dtype=torch.long, device=device)
    for row_index, row in enumerate(examples):
        for mention in row.mentions:
            value[row_index, mention.word_index] = mention.value
            phase[row_index, mention.word_index] = mention.phase
            address[row_index, mention.word_index] = mention.address
            pointer[row_index, mention.field] = mention.word_index
    if pointer.lt(0).any():
        raise ValueError("MQB1 supervisor lost a typed mention")
    states = torch.tensor(
        [(*row.before, *row.after) for row in examples],
        dtype=torch.long,
        device=device,
    )
    return words, mask, value, phase, address, pointer, states


def _balanced_word_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    none_class: int,
) -> torch.Tensor:
    loss = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
    positive = mask & target.ne(none_class)
    negative = mask & target.eq(none_class)
    if not positive.any() or not negative.any():
        raise ValueError("MQB1 batch lacks balanced mention supervision")
    return 0.5 * (loss[positive].mean() + loss[negative].mean())


def _pair_loss(logits: MentionBinderLogits, pointer: torch.Tensor) -> torch.Tensor:
    batch = pointer.shape[0]
    before = pointer[:, :REGISTER_COUNT]
    after = pointer[:, REGISTER_COUNT:]
    rows = torch.arange(batch, device=pointer.device)[:, None, None]
    pair = logits.pair[rows, before[:, :, None], after[:, None, :]]
    target = torch.eye(REGISTER_COUNT, device=pointer.device)[None].expand(batch, -1, -1)
    return F.binary_cross_entropy_with_logits(
        pair, target, pos_weight=torch.tensor(4.0, device=pointer.device)
    )


def _train_batch(
    model: MentionEvidenceBinder,
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
    words, mask, value, phase, address, pointer, states = _features_and_targets(
        source_model, examples, device
    )
    logits = model(words, mask)
    value_loss = _balanced_word_ce(logits.value, value, mask, NONE_VALUE)
    phase_loss = _balanced_word_ce(logits.phase, phase, mask, NONE_PHASE)
    address_loss = _balanced_word_ce(logits.address, address, mask, NONE_ADDRESS)
    pointer_logits = logits.field.transpose(1, 2).masked_fill(~mask[:, None], -torch.inf)
    pointer_loss = F.cross_entropy(
        pointer_logits.reshape(-1, pointer_logits.shape[-1]), pointer.reshape(-1)
    )
    pair_loss = _pair_loss(logits, pointer)
    loss = value_loss + phase_loss + address_loss + pointer_loss + pair_loss
    with torch.no_grad():
        rows = torch.arange(batch_size, device=device)[:, None]
        gold_value = logits.value.argmax(-1)[rows, pointer]
        gold_phase = logits.phase.argmax(-1)[rows, pointer]
        gold_address = logits.address.argmax(-1)[rows, pointer]
        expected_phase = torch.arange(FIELD_COUNT, device=device)[None] // REGISTER_COUNT
        expected_address = torch.arange(FIELD_COUNT, device=device)[None] % REGISTER_COUNT
        independent_pointer = pointer_logits.argmax(-1)
    return loss, {
        "loss": float(loss.detach()),
        "value_loss": float(value_loss.detach()),
        "phase_loss": float(phase_loss.detach()),
        "address_loss": float(address_loss.detach()),
        "pointer_loss": float(pointer_loss.detach()),
        "pair_loss": float(pair_loss.detach()),
        "gold_value_exact": float(gold_value.eq(states).float().mean()),
        "gold_phase_exact": float(gold_phase.eq(expected_phase).float().mean()),
        "gold_address_exact": float(gold_address.eq(expected_address).float().mean()),
        "independent_pointer_exact": float(independent_pointer.eq(pointer).float().mean()),
    }


def _reverse_valid(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    permutation = torch.arange(mask.shape[1], device=mask.device)[None].expand(
        mask.shape[0], -1
    ).clone()
    for row in range(mask.shape[0]):
        valid = torch.nonzero(mask[row], as_tuple=False).squeeze(-1)
        permutation[row, valid] = valid.flip(0)
    expansion = permutation
    while expansion.ndim < tensor.ndim:
        expansion = expansion.unsqueeze(-1)
    return tensor.gather(1, expansion.expand_as(tensor))


@torch.no_grad()
def evaluate_cohort(
    model: MentionEvidenceBinder,
    source_model,
    *,
    cohort: str,
    count: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    totals = {
        "valid": 0,
        "before": 0,
        "after": 0,
        "complete": 0,
        "assignment": 0,
        "value_fields": 0,
        "provenance_mismatches": 0,
        "accepted_duplicate_mentions": 0,
        "accepted_overflow": 0,
        "value_shuffle_complete": 0,
        "field_shuffle_complete": 0,
        "pair_negated_rejections": 0,
        "pair_negated_denominator": 0,
    }
    for start in range(0, count, batch_size):
        examples = [
            generate_mention_evidence(seed=seed + index, cohort=cohort)
            for index in range(start, min(count, start + batch_size))
        ]
        words, mask, _, _, _, pointer, states = _features_and_targets(
            source_model, examples, device
        )
        logits = model(words, mask)
        binding = model.decode(logits, mask)
        before_ok = binding.before.eq(states[:, :REGISTER_COUNT]).all(-1)
        after_ok = binding.after.eq(states[:, REGISTER_COUNT:]).all(-1)
        complete = binding.valid & before_ok & after_ok
        assignment = binding.valid & binding.provenance.eq(pointer).all(-1)
        totals["valid"] += int(binding.valid.sum())
        totals["before"] += int((binding.valid & before_ok).sum())
        totals["after"] += int((binding.valid & after_ok).sum())
        totals["complete"] += int(complete.sum())
        totals["assignment"] += int(assignment.sum())
        totals["value_fields"] += int(binding.selected_values.eq(states).sum())
        totals["provenance_mismatches"] += int(
            (binding.valid[:, None] & binding.provenance.ne(pointer)).sum()
        )
        sorted_words = binding.provenance.sort(-1).values
        duplicates = sorted_words[:, 1:].eq(sorted_words[:, :-1]).any(-1)
        totals["accepted_duplicate_mentions"] += int((binding.valid & duplicates).sum())
        totals["accepted_overflow"] += int((binding.valid & binding.overflow).sum())

        value_shuffled = model.decode(
            replace(logits, value=_reverse_valid(logits.value, mask)), mask
        )
        totals["value_shuffle_complete"] += int(
            (
                value_shuffled.valid
                & value_shuffled.before.eq(states[:, :REGISTER_COUNT]).all(-1)
                & value_shuffled.after.eq(states[:, REGISTER_COUNT:]).all(-1)
            ).sum()
        )
        field_shuffled = model.decode(
            replace(logits, field=_reverse_valid(logits.field, mask)), mask
        )
        totals["field_shuffle_complete"] += int(
            (
                field_shuffled.valid
                & field_shuffled.before.eq(states[:, :REGISTER_COUNT]).all(-1)
                & field_shuffled.after.eq(states[:, REGISTER_COUNT:]).all(-1)
            ).sum()
        )

        pair = logits.pair.clone()
        rows = torch.arange(len(examples), device=device)[:, None]
        before_words = binding.provenance[:, :REGISTER_COUNT]
        after_words = binding.provenance[:, REGISTER_COUNT:]
        pair[rows, before_words, after_words] = -pair[
            rows, before_words, after_words
        ].abs() - 1
        pair[rows, after_words, before_words] = -pair[
            rows, after_words, before_words
        ].abs() - 1
        pair_negated = model.decode(replace(logits, pair=pair), mask)
        totals["pair_negated_rejections"] += int(
            (binding.valid & ~pair_negated.valid).sum()
        )
        totals["pair_negated_denominator"] += int(binding.valid.sum())

    denominator = max(1, totals["pair_negated_denominator"])
    return {
        "examples": count,
        "valid_rate": totals["valid"] / count,
        "before_state_exact": totals["before"] / count,
        "after_state_exact": totals["after"] / count,
        "complete_state_pair_exact": totals["complete"] / count,
        "complete_assignment_exact": totals["assignment"] / count,
        "selected_value_exact": totals["value_fields"] / (count * FIELD_COUNT),
        "provenance_mismatches": totals["provenance_mismatches"],
        "accepted_duplicate_mentions": totals["accepted_duplicate_mentions"],
        "accepted_overflow": totals["accepted_overflow"],
        "value_shuffle_complete": totals["value_shuffle_complete"] / count,
        "field_shuffle_complete": totals["field_shuffle_complete"] / count,
        "pair_negated_rejection": totals["pair_negated_rejections"] / denominator,
        "pair_negated_denominator": totals["pair_negated_denominator"],
    }


def renderer_parity(count: int) -> dict[str, object]:
    mismatches = 0
    represented = 0
    distractor_labels = 0
    per_cohort = count // len(EVIDENCE_COHORTS)
    for cohort_index, cohort in enumerate(EVIDENCE_COHORTS):
        for index in range(per_cohort):
            seed = SEED + 70_000_000 + COHORT_OFFSETS[cohort] + index
            old = generate_probe_evidence(seed=seed, cohort=cohort)
            new = generate_mention_evidence(seed=seed, cohort=cohort)
            mismatches += int(old.words != new.words)
            represented += len(new.mentions)
            distractor_labels += max(0, len(new.mentions) - FIELD_COUNT)
    examples = per_cohort * len(EVIDENCE_COHORTS)
    return {
        "examples": examples,
        "word_mismatches": mismatches,
        "represented_mentions": represented,
        "expected_mentions": examples * FIELD_COUNT,
        "distractor_field_labels": distractor_labels,
        "pass": mismatches == 0
        and represented == examples * FIELD_COUNT
        and distractor_labels == 0,
    }


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
    mei1, mei1_report, frozen_before = _load_qualified_mei1(
        args.mei1_checkpoint, device
    )
    config = MQB1Config(
        input_width=args.input_width,
        width=args.width,
        heads=args.heads,
        layers=args.layers,
        pair_width=args.pair_width,
    )
    model = MentionEvidenceBinder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    parity = renderer_parity(args.parity_count)
    if not parity["pass"]:
        raise ValueError("MQB1 assessor rendering differs from MEI1")
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
        )
        for cohort in EVIDENCE_COHORTS
    }
    frozen_after = {
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
    integrity_counts = sum(
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
        "zero_integrity_failures": integrity_counts == 0,
        "value_shuffle_drops_shifted_50pp_each": min(
            row["complete_state_pair_exact"] - row["value_shuffle_complete"]
            for row in shifted
        ) >= 0.50,
        "field_shuffle_drops_shifted_50pp_each": min(
            row["complete_state_pair_exact"] - row["field_shuffle_complete"]
            for row in shifted
        ) >= 0.50,
        "pair_negation_rejects_100pct_each": min(
            row["pair_negated_rejection"] for row in evaluations.values()
        ) >= 1.0,
        "frozen_algebra_query_unchanged": frozen_before == frozen_after,
        "candidate_source_audit": bool(architecture_receipt(model)["source_audit"]["pass"]),
    }
    gate["pass"] = all(gate.values())
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    elapsed = time.time() - started
    report = {
        "schema": SCHEMA,
        "status": "bounded-structural-mention-gate-complete",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "architecture": architecture_receipt(model),
        "inputs": {
            "base_sha256": _sha256(args.base),
            "tokenizer_sha256": _sha256(args.tokenizer),
            "sc1_sha256": _sha256(args.sc1_checkpoint),
            "hsc1_sha256": _sha256(args.hsc1_checkpoint),
            "mei1_sha256": _sha256(args.mei1_checkpoint),
            "mei1_component_gate_passed": bool(mei1_report["gate"]["pass"]),
            "mei1_executor_query_qualified": True,
        },
        "renderer_parity": parity,
        "training": {
            "elapsed_seconds": elapsed,
            "examples": args.updates * args.batch_size,
            "examples_per_second": args.updates * args.batch_size / elapsed,
            "updates": args.updates,
            "final_batch": last,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated())
            if device.type == "cuda"
            else 0,
            "model_state_sha256": _state_sha256(state),
        },
        "frozen_component_hashes_before": frozen_before,
        "frozen_component_hashes_after": frozen_after,
        "evaluations": evaluations,
        "gate": gate,
        "claim_boundary": (
            "Synthetic structural evidence-binding component gate only. Full DIVERGE "
            "composition remains blocked unless every conjunctive gate passes."
        ),
    }
    checkpoint = {
        "schema": SCHEMA,
        "config": asdict(config),
        "state_dict": state,
        "model_state_sha256": report["training"]["model_state_sha256"],
        "frozen_mei1_sha256": report["inputs"]["mei1_sha256"],
        "frozen_component_hashes": frozen_after,
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
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-count", type=int, default=20000)
    parser.add_argument("--parity-count", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
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
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--pair-width", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite MQB1 output")
    if args.seed != SEED or args.updates != 1600 or args.batch_size != 64:
        raise ValueError("MQB1 training contract differs from the frozen gate")
    if args.eval_count != 20000 or args.parity_count != 10000:
        raise ValueError("MQB1 evaluation contract differs from the frozen gate")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    report = run(args)
    if not report["gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
