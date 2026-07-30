#!/usr/bin/env python3
"""Localize ETTR learning at oracle-separated component interfaces.

This read-only diagnostic asks three questions on the development split:

1. Can the WORLD compiler emit the supervised initial packet?
2. Can the COMMAND reactor choose the next transaction when every prior
   state transition is teacher-forced?
3. Can the QUERY reader answer matched causal queries from the exact
   supervised terminal packet?

The oracle packets are used only inside this assessor-side diagnostic. They
never enter a candidate checkpoint or the autonomous evaluation path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TRANSACTION_COUNT,
    TheoryReactorConfig,
    TransactionPolicy,
    TypedTheoryState,
)
from ettr_checkpoint import load_ettr_checkpoint
from ettr_data_contract import ETTRContinuationBatch
from ettr_objectives import (
    ETTRCausalQueryPair,
    ETTRObjectiveConfig,
    ETTRPacketTargets,
    ETTRTransactionTargets,
)
from ettr_optimization import ETTROptimizerBundle
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_v3 import (
    ETTRV3EvaluationError,
    _HEX40,
    _HEX64,
    _build_model,
    _parameter_sha256,
    _read_hash_bound_json,
    _sha256_file,
    _validate_checkpoint_cursor,
    _validate_run_contract,
    _write_no_replace,
)
from probe_ettr_causal_queries import _pair_rows, _summary


REPORT_SCHEMA = "shohin-ettr-il-v3-oracle-interface-probe-v1"
_PACKET_FIELDS = (
    "active",
    "root",
    "value_code",
    "type_index",
    "relations",
    "committed",
    "halted",
)
_POLICY_FIELDS = (
    "opcode",
    "source",
    "target",
    "relation",
    "type_index",
    "value_code",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--run-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--architecture-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--max-batches", type=int, default=64)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if (
        _HEX64.fullmatch(args.release_sha256) is None
        or _HEX64.fullmatch(args.checkpoint_sha256) is None
        or _HEX64.fullmatch(args.run_contract_sha256) is None
        or _HEX40.fullmatch(args.source_commit) is None
        or not 0 <= args.architecture_seed < 2**63
        or not 0 <= args.data_seed < 2**63
        or args.max_batches < 2
    ):
        raise ETTRV3EvaluationError("ETTR oracle-interface arguments differ")


def packet_targets_to_state(
    targets: ETTRPacketTargets,
    config: TheoryReactorConfig,
    *,
    step: int,
    dtype: torch.dtype,
) -> TypedTheoryState:
    """Convert assessor labels to an exact categorical state."""

    active = targets.active.to(dtype)
    return TypedTheoryState(
        value_probabilities=(
            F.one_hot(targets.value_code, config.num_value_codes).to(dtype)
            * active.unsqueeze(-1)
        ),
        type_probabilities=(
            F.one_hot(targets.type_index, config.num_types).to(dtype)
            * active.unsqueeze(-1)
        ),
        relations=targets.relations.to(dtype),
        active=active,
        root=targets.root.to(dtype),
        committed=targets.committed.to(dtype),
        halted=targets.halted.to(dtype),
        step=step,
    )


def target_policy(
    targets: ETTRTransactionTargets,
    config: TheoryReactorConfig,
    step: int,
    *,
    dtype: torch.dtype,
) -> TransactionPolicy:
    """Build the exact generic transaction selected by offline labels."""

    values = {
        "opcode": F.one_hot(
            targets.opcode[:, step],
            TRANSACTION_COUNT,
        ).to(dtype),
        "source": F.one_hot(
            targets.source[:, step],
            config.num_slots,
        ).to(dtype),
        "target": F.one_hot(
            targets.target[:, step],
            config.num_slots,
        ).to(dtype),
        "relation": F.one_hot(
            targets.relation[:, step],
            config.num_relations,
        ).to(dtype),
        "type_index": F.one_hot(
            targets.type_index[:, step],
            config.num_types,
        ).to(dtype),
        "value_code": F.one_hot(
            targets.value_code[:, step],
            config.num_value_codes,
        ).to(dtype),
    }
    return TransactionPolicy(
        **values,
        opcode_probabilities=values["opcode"],
        source_probabilities=values["source"],
        target_probabilities=values["target"],
        relation_probabilities=values["relation"],
        type_probabilities=values["type_index"],
        value_probabilities=values["value_code"],
    )


def policy_masks(
    targets: ETTRTransactionTargets,
) -> Mapping[str, torch.Tensor]:
    valid = targets.step_mask
    opcode = targets.opcode
    return {
        "opcode": valid,
        "source": valid & (opcode <= 5),
        "target": valid & ((opcode == 3) | (opcode == 4)),
        "relation": valid & ((opcode == 3) | (opcode == 4)),
        "type_index": valid & (opcode == 0),
        "value_code": valid & ((opcode == 0) | (opcode == 1)),
    }


def _packet_batch_counts(
    state: TypedTheoryState,
    targets: ETTRPacketTargets,
) -> dict[str, tuple[int, int]]:
    categorical = targets.slot_mask & targets.active
    masks = {
        "active": targets.slot_mask,
        "root": targets.slot_mask,
        "value_code": categorical,
        "type_index": categorical,
        "relations": targets.relation_mask,
        "committed": torch.ones_like(targets.committed),
        "halted": torch.ones_like(targets.halted),
    }
    predictions = {
        "active": state.active.ge(0.5),
        "root": state.root.ge(0.5),
        "value_code": state.value_probabilities.argmax(-1),
        "type_index": state.type_probabilities.argmax(-1),
        "relations": state.relations.ge(0.5),
        "committed": state.committed.ge(0.5),
        "halted": state.halted.ge(0.5),
    }
    counts: dict[str, tuple[int, int]] = {}
    row_correct = torch.ones(
        targets.active.shape[0],
        dtype=torch.bool,
        device=targets.active.device,
    )
    for name in _PACKET_FIELDS:
        mask = masks[name]
        correct = predictions[name].eq(getattr(targets, name))
        counts[name] = (
            int((correct & mask).sum().detach().cpu()),
            int(mask.sum().detach().cpu()),
        )
        row_correct &= (correct | ~mask).reshape(correct.shape[0], -1).all(-1)
    counts["joint"] = (
        int(row_correct.sum().detach().cpu()),
        row_correct.numel(),
    )
    return counts


def _teacher_forced_policy_counts(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
) -> dict[str, tuple[int, int]]:
    targets = batch.transaction_targets
    masks = policy_masks(targets)
    state = packet_targets_to_state(
        batch.packet_targets,
        model.config,
        step=0,
        dtype=next(model.reactor.parameters()).dtype,
    )
    command_hidden = model._encode_to_stage(
        batch.episodes.command.tokens,
        pos=0,
    )
    counts = {name: [0, 0] for name in (*_POLICY_FIELDS, "joint")}
    for step in range(targets.opcode.shape[1]):
        predicted = model.reactor.policy(
            state,
            hard=True,
            command_hidden=command_hidden,
            command_attention_mask=batch.episodes.command.attention_mask,
            validate=False,
        )
        valid = targets.step_mask[:, step]
        joint = torch.ones_like(valid)
        for name in _POLICY_FIELDS:
            mask = masks[name][:, step]
            correct = getattr(predicted, name).argmax(-1).eq(
                getattr(targets, name)[:, step]
            )
            counts[name][0] += int((correct & mask).sum().detach().cpu())
            counts[name][1] += int(mask.sum().detach().cpu())
            joint &= correct | ~mask
        counts["joint"][0] += int((joint & valid).sum().detach().cpu())
        counts["joint"][1] += int(valid.sum().detach().cpu())
        state = model.reactor.apply(
            state,
            target_policy(
                targets,
                model.config,
                step,
                dtype=state.active.dtype,
            ),
            hard=True,
            validate=False,
        )
    return {name: tuple(value) for name, value in counts.items()}


def _gather_query_logits(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
    state: TypedTheoryState,
) -> torch.Tensor:
    logits, _ = model.answer_query(
        state,
        batch.episodes.query.tokens,
        attention_mask=batch.episodes.query.attention_mask,
    )
    return logits.gather(
        1,
        batch.episodes.query_read_index[:, None, None].expand(
            -1,
            1,
            logits.shape[-1],
        ),
    ).squeeze(1)


def _oracle_reader_pairs(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
) -> Mapping[str, ETTRCausalQueryPair]:
    oracle = packet_targets_to_state(
        batch.terminal_packet_targets,
        model.config,
        step=batch.transaction_targets.opcode.shape[1],
        dtype=next(model.query_reader.parameters()).dtype,
    )
    logits = _gather_query_logits(model, batch, oracle)
    (
        _world_packet,
        world_command,
        world_target,
        command_packet,
        _command_command,
        command_target,
    ) = batch.causal_rectangles.intervention_indices()
    targets = batch.episodes.query.targets.gather(
        1,
        batch.episodes.query_read_index[:, None],
    ).squeeze(1)
    return {
        "world": ETTRCausalQueryPair(
            correct_logits=logits.index_select(0, world_target),
            foil_logits=logits.index_select(0, world_command),
            correct_target=targets.index_select(0, world_target),
            foil_target=targets.index_select(0, world_command),
        ),
        "command": ETTRCausalQueryPair(
            correct_logits=logits.index_select(0, command_target),
            foil_logits=logits.index_select(0, command_packet),
            correct_target=targets.index_select(0, command_target),
            foil_target=targets.index_select(0, command_packet),
        ),
    }


def _merge_counts(
    destination: dict[str, list[int]],
    source: Mapping[str, tuple[int, int]],
) -> None:
    for name, (correct, total) in source.items():
        destination.setdefault(name, [0, 0])
        destination[name][0] += correct
        destination[name][1] += total


def _count_summary(
    values: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    return {
        name: {
            "accuracy": None if total == 0 else correct / total,
            "correct": correct,
            "total": total,
        }
        for name, (correct, total) in sorted(values.items())
    }


def _arm_batch(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
) -> tuple[
    dict[str, tuple[int, int]],
    dict[str, tuple[int, int]],
    Mapping[str, list[dict[str, object]]],
]:
    world_hidden = model._encode_to_stage(
        batch.episodes.world.tokens,
        pos=0,
    )
    compiler_state = model.compiler(
        world_hidden,
        attention_mask=batch.episodes.world.attention_mask,
        hard=True,
    )
    reader_pairs = _oracle_reader_pairs(model, batch)
    return (
        _packet_batch_counts(compiler_state, batch.packet_targets),
        _teacher_forced_policy_counts(model, batch),
        {
            kind: _pair_rows(pair)
            for kind, pair in reader_pairs.items()
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise ETTRV3EvaluationError(
            "ETTR oracle-interface probe requires CUDA"
        )
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRV3EvaluationError(
            "ETTR oracle-interface probe requires H100"
        )
    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    run_contract = _read_hash_bound_json(
        args.run_contract,
        expected_sha256=args.run_contract_sha256,
        label="ETTR run contract",
    )
    model_config, optimizer_config = _validate_run_contract(
        run_contract,
        release_sha256=args.release_sha256,
        release_source_commit=stream.release["source_commit"],
        architecture_seed=args.architecture_seed,
    )
    models = {}
    raw_model, raw_provenance = _build_model(
        args.protected_checkpoint,
        architecture_seed=args.architecture_seed,
        model_config=model_config,
        device=device,
    )
    models["raw"] = raw_model
    checkpoint_model, checkpoint_provenance = _build_model(
        args.protected_checkpoint,
        architecture_seed=args.architecture_seed,
        model_config=model_config,
        device=device,
    )
    optimizer = ETTROptimizerBundle(checkpoint_model, optimizer_config)
    resumed = load_ettr_checkpoint(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        model=checkpoint_model,
        protected_base=checkpoint_provenance,
        optimizer=optimizer,
        scheduler=None,
    )
    _validate_checkpoint_cursor(
        resumed.progress,
        resumed.data_stream,
        run_contract=run_contract,
        stream=stream,
        release_sha256=args.release_sha256,
        protected_step=checkpoint_provenance.step,
    )
    del optimizer
    if (
        raw_provenance.checkpoint_sha256
        != stream.manifest.protected_checkpoint_sha256
        or run_contract["parameter_receipt"]
        != asdict(raw_model.parameter_receipt())
    ):
        raise ETTRV3EvaluationError(
            "ETTR oracle-interface provenance differs"
        )
    for model in models.values():
        model.eval()

    counts = {
        arm: {
            "compiler": {},
            "teacher_forced_reactor": {},
        }
        for arm in models
    }
    reader_rows = {
        arm: {"world": [], "command": []}
        for arm in models
    }
    packet_index = ETTRDiskPacketSufficiencyIndex(
        stream.packet_index_root
    )
    batches = 0
    try:
        iterator = stream.iter_positioned_batches(
            "development",
            rank=0,
            world_size=1,
            epoch=0,
            seed=args.data_seed,
        )
        for _, cpu_batch in iterator:
            if batches >= args.max_batches:
                break
            packet_index.verify_validation((cpu_batch,))
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(
                raw_model.config,
                ETTRObjectiveConfig(
                    vocab_size=raw_model.base.cfg.vocab_size
                ),
            )
            for arm, model in models.items():
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    compiler, reactor, reader = _arm_batch(model, batch)
                _merge_counts(counts[arm]["compiler"], compiler)
                _merge_counts(
                    counts[arm]["teacher_forced_reactor"],
                    reactor,
                )
                for kind, values in reader.items():
                    reader_rows[arm][kind].extend(values)
            batches += 1
    finally:
        packet_index.close()
    if batches != args.max_batches:
        raise ETTRV3EvaluationError(
            "ETTR oracle-interface development split is too short"
        )

    report = {
        "architecture_seed": args.architecture_seed,
        "arms": {
            arm: {
                "compiler": _count_summary(values["compiler"]),
                "oracle_terminal_reader": {
                    kind: _summary(rows)
                    for kind, rows in reader_rows[arm].items()
                },
                "teacher_forced_reactor": _count_summary(
                    values["teacher_forced_reactor"]
                ),
            }
            for arm, values in counts.items()
        },
        "assessor_oracle_boundary": {
            "candidate_checkpoint_modified": False,
            "oracle_at_autonomous_inference": False,
            "oracle_state_used_for": [
                "teacher_forced_reactor_diagnostic",
                "terminal_reader_diagnostic",
            ],
        },
        "batches": batches,
        "checkpoint": {
            "parameter_sha256": _parameter_sha256(checkpoint_model),
            "progress": asdict(resumed.progress),
            "run_contract_sha256": args.run_contract_sha256,
            "sha256": args.checkpoint_sha256,
        },
        "data_seed": args.data_seed,
        "device": {
            "bf16": torch.cuda.is_bf16_supported(),
            "name": torch.cuda.get_device_name(device),
        },
        "protected_checkpoint_sha256": raw_provenance.checkpoint_sha256,
        "raw_parameter_sha256": _parameter_sha256(raw_model),
        "release_file_sha256": args.release_sha256,
        "release_manifest_sha256": stream.manifest.sha256(),
        "schema": REPORT_SCHEMA,
        "source_commit": args.source_commit,
        "source_verification": source_verification,
        "split": "development",
        "tokenizer_sha256": _sha256_file(args.tokenizer),
    }
    payload = (
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    digest = _write_no_replace(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
