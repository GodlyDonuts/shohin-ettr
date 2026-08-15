#!/usr/bin/env python3
"""Generate matched source-disjoint arms for an upward-MoE temporal gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from hf_pcf1_evaluate import self_refinement_prompt, shard_bounds
from hf_product_reasoning_eval import (
    GENERATED_ONLY_SEQUENCE_CONTRACT,
    _generate_completions,
    _generation_stop_token_ids,
    _render_prompt,
)
from hf_q36_mtr_evaluate import q36_nonpadding_prompt_tokens
from hf_upward_moe_train_owner import (
    UpwardMoEOwnerTrainingError,
    _load_host,
)
from hf_upward_moe_train_temporal_gate import (
    GATE_CAUSAL_LOSS_WEIGHT,
    GATE_INITIAL_REVISION_WEIGHT,
    GATE_ROUTING_SUPERVISION_WEIGHT,
    UpwardMoETemporalTrainingError,
    host_spec,
    restore_gate_checkpoint,
)
from upward_moe_role_lineage import (
    UpwardMoERoleLineageError,
    load_role_checkpoint,
    load_role_pair,
    sha256_file,
)
from upward_moe_temporal_gate import (
    MixtralTemporalGateModel,
    NemotronSuperTemporalGateModel,
    UpwardMoETemporalGateError,
)

DATA_SCHEMA = "shohin-q36-mtr-eval-v1"
CANDIDATE_SCHEMA = "shohin-upward-moe-temporal-candidate-v1"
REPORT_SCHEMA = "shohin-upward-moe-temporal-evaluation-v1"
ARMS = ("unchanged", "self_refinement", "owner", "aligned_revision", "temporal_gate")
ROWS = 1_289
SHARDS = 16
SEED = 2026080816
MAX_NEW_TOKENS = 768


class UpwardMoETemporalEvaluationError(RuntimeError):
    """The upward-MoE temporal evaluation contract differed."""


def load_evaluation_rows(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise UpwardMoETemporalEvaluationError(
            "upward temporal evaluation bytes differ"
        )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    identities: set[str] = set()
    for row in rows:
        identity = row.get("identity_sha256") if isinstance(row, dict) else None
        draft = row.get("internal_draft") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("schema") != DATA_SCHEMA
            or row.get("split") != "development"
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in identities
            or not isinstance(row.get("task"), str)
            or not isinstance(row.get("source_prompt"), str)
            or not row["source_prompt"].strip()
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(draft, dict)
            or draft.get("identity_sha256") != identity
            or not isinstance(draft.get("completion"), str)
            or not draft["completion"].strip()
            or row.get("internal_draft_visible") is not True
            or row.get("external_candidate_text_visible") is not False
            or any(key in row for key in ("assessor", "answer", "response", "gold"))
        ):
            raise UpwardMoETemporalEvaluationError(
                "upward temporal evaluation row differs"
            )
        identities.add(identity)
    if len(rows) != ROWS:
        raise UpwardMoETemporalEvaluationError(
            "upward temporal evaluation population differs"
        )
    return sorted(rows, key=lambda row: row["identity_sha256"])


def prompt_for(arm: str, row: dict[str, Any]) -> str:
    if arm in {"owner", "unchanged"}:
        return str(row["source_prompt"])
    if arm == "self_refinement":
        return self_refinement_prompt(row)
    if arm in {"aligned_revision", "temporal_gate"}:
        return str(row["question"])
    raise UpwardMoETemporalEvaluationError("upward temporal arm differs")


def restore_role(model: Any, checkpoint: Path, spec: Any, role: str) -> dict[str, Any]:
    try:
        payload = load_role_checkpoint(checkpoint, spec)
    except UpwardMoERoleLineageError as error:
        raise UpwardMoETemporalEvaluationError(str(error)) from error
    metadata = payload["metadata"]
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if metadata["role"] != role or set(current) != set(payload["trainable_state"]):
        raise UpwardMoETemporalEvaluationError("upward temporal role differs")
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(payload["trainable_state"][name].to(parameter.device))
    if model.trainable_state_sha256() != metadata["final_trainable_state_sha256"]:
        raise UpwardMoETemporalEvaluationError("upward temporal role restore differs")
    return {
        "role": role,
        "checkpoint_sha256": sha256_file(checkpoint),
        "state_sha256": metadata["final_trainable_state_sha256"],
        "restore_exact": True,
    }


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise UpwardMoETemporalEvaluationError("upward temporal candidates exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise UpwardMoETemporalEvaluationError("upward temporal report exists")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec = host_spec(args.host)
    if (
        args.arm not in ARMS
        or args.seed != SEED
        or args.expected_rows != ROWS
        or args.shard_count != SHARDS
        or args.batch_size != 1
        or not 0 <= args.shard_index < SHARDS
        or args.candidates_output.exists()
        or args.report.exists()
    ):
        raise UpwardMoETemporalEvaluationError("upward temporal settings differ")
    rows = load_evaluation_rows(args.data, args.expected_data_sha256)
    start, end = shard_bounds(ROWS, args.shard_index, SHARDS, args.batch_size)
    rows = rows[start:end]
    try:
        owner_state, revision_state, lineage = load_role_pair(
            args.owner_checkpoint, args.revision_checkpoint, spec
        )
    except UpwardMoERoleLineageError as error:
        raise UpwardMoETemporalEvaluationError(str(error)) from error
    role_checkpoint = (
        args.owner_checkpoint
        if args.arm == "owner"
        else args.revision_checkpoint if args.arm == "aligned_revision" else None
    )
    attach_revision = role_checkpoint is not None
    try:
        loaded = _load_host(args, attach_revision=attach_revision)
    except UpwardMoEOwnerTrainingError as error:
        raise UpwardMoETemporalEvaluationError(str(error)) from error
    model_receipt: dict[str, Any] = {
        "host_contract": spec.receipt(),
        "model_receipt": loaded.model_receipt,
        "role_lineage": lineage,
    }
    if role_checkpoint is not None:
        model = loaded.model
        model_receipt["role"] = restore_role(
            model,
            role_checkpoint,
            spec,
            "owner" if args.arm == "owner" else "aligned",
        )
        generator = model.backbone
        model.eval()
        model.reset_receipt()
    elif args.arm == "temporal_gate":
        if args.gate_checkpoint is None:
            raise UpwardMoETemporalEvaluationError("temporal checkpoint is absent")
        try:
            model = (
                NemotronSuperTemporalGateModel(
                    loaded.model, owner_state, revision_state
                )
                if args.host == "nemotron-super"
                else MixtralTemporalGateModel(loaded.model, owner_state, revision_state)
            )
            metadata = restore_gate_checkpoint(args.gate_checkpoint, model, spec)
        except (
            UpwardMoERoleLineageError,
            UpwardMoETemporalGateError,
            UpwardMoETemporalTrainingError,
        ) as error:
            raise UpwardMoETemporalEvaluationError(str(error)) from error
        expected = {
            "architecture": spec.architecture,
            "host_contract": spec.receipt(),
            "causal_loss_weight": GATE_CAUSAL_LOSS_WEIGHT,
            "routing_supervision_weight": GATE_ROUTING_SUPERVISION_WEIGHT,
            "initial_revision_weight": GATE_INITIAL_REVISION_WEIGHT,
            "role_receipt": lineage,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise UpwardMoETemporalEvaluationError(
                "upward temporal gate metadata differs"
            )
        model_receipt["temporal_gate"] = {
            "checkpoint_sha256": sha256_file(args.gate_checkpoint),
            "metadata_sha256": hashlib.sha256(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "lineage": lineage,
        }
        generator = model.backbone
        model.eval()
        model.reset_receipt()
    else:
        model = None
        generator = loaded.model
        generator.eval()
    tokenizer = loaded.tokenizer
    tokenizer.padding_side = "left"
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    for index in range(2):
        torch.cuda.reset_peak_memory_stats(index)
    counters: Counter[str] = Counter()
    candidates = []
    started = time.monotonic()
    for row in rows:
        rendered = [_render_prompt(tokenizer, prompt_for(args.arm, row), True, False)]
        counters["prompt_tokens"] += q36_nonpadding_prompt_tokens(tokenizer, rendered)
        completions, usage = _generate_completions(
            generator,
            tokenizer,
            rendered,
            False,
            "greedy",
            MAX_NEW_TOKENS,
            stop_ids,
            add_special_tokens=False,
        )
        completion = completions[0]
        generated_tokens, exhausted = usage[0]
        candidates.append(
            {
                "schema": CANDIDATE_SCHEMA,
                "host": spec.host,
                "arm": args.arm,
                "identity_sha256": row["identity_sha256"],
                "task": row["task"],
                "completion": completion,
                "generated_tokens": generated_tokens,
                "max_token_exhausted": exhausted,
            }
        )
        counters["rows"] += 1
        counters["generated_tokens"] += generated_tokens
        counters["max_token_exhausted"] += int(exhausted)
        counters["empty_completions"] += int(not completion.strip())
    torch.cuda.synchronize()
    candidates_sha256 = _atomic_lines(args.candidates_output, candidates)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "host": spec.host,
        "arm": args.arm,
        "split": "development",
        "data_sha256": args.expected_data_sha256,
        "mechanics_report_sha256": sha256_file(args.mechanics_report),
        **model_receipt,
        "generation_mode": "greedy",
        "generation_sequence_contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": start,
        "row_end": end,
        "full_row_count": ROWS,
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidates_sha256,
        "counters": dict(sorted(counters.items())),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in range(2)
        },
        "routing_receipt": model.receipt() if model is not None else None,
        "assessor_access_count": 0,
        "development_labels_read": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", choices=("nemotron-super", "mixtral-8x22b"), required=True
    )
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--expected-model-manifest-sha256")
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--overlay-manifest", type=Path)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--revision-checkpoint", type=Path, required=True)
    parser.add_argument("--gate-checkpoint", type=Path)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=ROWS)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=SHARDS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
