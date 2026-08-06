"""Exact assessor for persistent discrete state replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import torch

from diverge_rsm1_data import (
    RSM1DataError,
    ReplayTokens,
    build_replay_supervision,
    decode_state,
    tokenize_replay_example,
)
from diverge_rsm1_product import (
    RSM1ProductModel,
    load_rsm1_checkpoint,
    module_state_sha256,
)
from hf_product_reasoning_train import load_product_backbone


BOARD_SCHEMA = "shohin-diverge-crp1-board-v1"
REPORT_SCHEMA = "shohin-diverge-rsm1-evaluation-v1"


class RSM1EvalError(RuntimeError):
    """The persistent replay evaluation contract was violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RSM1EvalError(f"refusing to replace evaluation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _read_board(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != BOARD_SCHEMA or row.get("split") != "evaluation":
            raise RSM1EvalError("RSM1 evaluation board differs")
        identity = str(row.get("identity_sha256") or "")
        if len(identity) != 64 or identity in identities:
            raise RSM1EvalError("RSM1 evaluation identity differs")
        identities.add(identity)
        rows.append(row)
    if len(rows) != 480:
        raise RSM1EvalError("RSM1 evaluation board must contain 480 rows")
    return rows


def _tokenize(
    tokenizer: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> ReplayTokens:
    if args.trace_kind == "wrong":
        steps = row["wrong_steps"]
        answer = row["wrong_answer"]
    else:
        steps = row["correct_steps"]
        answer = row["answer"]
    tokens = tokenize_replay_example(
        tokenizer,
        row,
        steps,
        f"Final answer: \\boxed{{{answer}}}",
        max_sequence_length=args.max_sequence_length,
        packet_slots=args.packet_slots,
    )
    if tokens is None:
        raise RSM1EvalError("admitted RSM1 evaluation row no longer fits")
    return tokens


def _family_valid(family: str, value: str) -> bool:
    if family == "scalar":
        return re.fullmatch(r"-?\d+", value) is not None
    if family == "register":
        return re.fullmatch(r"-?\d+,-?\d+", value) is not None
    if family == "symbolic":
        return re.fullmatch(r"[a-z]+", value) is not None
    raise RSM1EvalError("unknown RSM1 family")


def _decode(tokens: list[int], family: str) -> tuple[str | None, bool]:
    try:
        value = decode_state(tokens)
    except RSM1DataError:
        return None, False
    return value, _family_valid(family, value)


def _empty_counts() -> dict[str, int]:
    return {
        "rows": 0,
        "terminal_correct": 0,
        "packet_correct": 0,
        "initial_state_correct": 0,
        "full_trajectory_correct": 0,
        "valid_terminal": 0,
        "invalid_terminal": 0,
        "active_transitions": 0,
        "exact_transitions": 0,
    }


def _update_counts(
    counts: dict[str, int],
    *,
    terminal_correct: bool,
    packet_correct: bool,
    initial_correct: bool,
    trajectory_correct: bool,
    valid_terminal: bool,
    active_transitions: int,
    exact_transitions: int,
) -> None:
    counts["rows"] += 1
    counts["terminal_correct"] += int(terminal_correct)
    counts["packet_correct"] += int(packet_correct)
    counts["initial_state_correct"] += int(initial_correct)
    counts["full_trajectory_correct"] += int(trajectory_correct)
    counts["valid_terminal"] += int(valid_terminal)
    counts["invalid_terminal"] += int(not valid_terminal)
    counts["active_transitions"] += active_transitions
    counts["exact_transitions"] += exact_transitions


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise RSM1EvalError(f"output already exists: {args.output}")
    expected_files = (
        (args.data, args.data_sha256, "evaluation board"),
        (args.source_checkpoint, args.source_checkpoint_sha256, "source checkpoint"),
        (args.crp_checkpoint, args.crp_checkpoint_sha256, "CRP1 checkpoint"),
        (args.rsm_checkpoint, args.rsm_checkpoint_sha256, "RSM1 checkpoint"),
    )
    for path, expected, label in expected_files:
        if _sha256_file(path) != expected:
            raise RSM1EvalError(f"RSM1 {label} hash differs")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RSM1EvalError("RSM1 requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backbone, resolved_loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = RSM1ProductModel(
        backbone,
        args.source_checkpoint,
        args.crp_checkpoint,
        source_checkpoint_sha256=args.source_checkpoint_sha256,
        crp_checkpoint_sha256=args.crp_checkpoint_sha256,
        source_revision=args.model_revision,
        packet_arm=args.packet_arm,
        state_width=args.state_width,
        state_slots=args.state_slots,
        packet_slots=args.packet_slots,
        max_trace_steps=args.max_trace_steps,
        attention_heads=args.attention_heads,
        ff_multiplier=args.ff_multiplier,
    ).to("cuda:0")
    update, metadata = load_rsm1_checkpoint(args.rsm_checkpoint, model)
    expected_metadata = {
        "architecture": "diverge-rsm1",
        "packet_arm": args.packet_arm,
        "model_revision": args.model_revision,
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "crp_checkpoint_sha256": args.crp_checkpoint_sha256,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches or update != args.expected_update:
        raise RSM1EvalError(
            f"RSM1 checkpoint metadata differs: {mismatches}, update={update}"
        )
    expected_replay_config = {
        "backbone_width": model.replay_config.backbone_width,
        "state_width": args.state_width,
        "state_slots": args.state_slots,
        "packet_slots": args.packet_slots,
        "max_trace_steps": args.max_trace_steps,
        "attention_heads": args.attention_heads,
        "ff_multiplier": args.ff_multiplier,
        "state_vocab_size": model.replay_config.state_vocab_size,
    }
    if metadata.get("replay_config") != expected_replay_config:
        raise RSM1EvalError("RSM1 checkpoint replay geometry differs")
    model.eval()
    model.set_ablation(args.ablation)
    frozen_initial = model.frozen_crp_sha256()
    replay_initial = module_state_sha256(model.replay)
    rows = _read_board(args.data)
    overall = _empty_counts()
    families = {family: _empty_counts() for family in ("scalar", "register", "symbolic")}
    results: list[dict[str, Any]] = []

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        tokens = [_tokenize(tokenizer, row, args) for row in batch_rows]
        target_selection = [
            int(row["error_index"]) if args.trace_kind == "wrong" else 0
            for row in batch_rows
        ]
        forced = (
            torch.tensor(target_selection, device="cuda:0", dtype=torch.long)
            if args.selection_mode == "forced"
            else None
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference = model.run_replay(
                [value.prompt_ids for value in tokens],
                [value.problem_mask for value in tokens],
                [value.packet_step_masks for value in tokens],
                [value.operation_masks for value in tokens],
                [value.final_mask for value in tokens],
                tokenizer.pad_token_id,
                selection_targets=forced,
            )
        selected = inference.selected_candidates.cpu().tolist()
        traces = inference.replay.state_trace_tokens.cpu().tolist()
        terminals = inference.replay.terminal_tokens.cpu().tolist()
        for row, expected_selection, actual_selection, trace, terminal in zip(
            batch_rows,
            target_selection,
            selected,
            traces,
            terminals,
            strict=True,
        ):
            family = str(row["family"])
            supervision = build_replay_supervision(
                row,
                int(actual_selection),
                max_trace_steps=args.max_trace_steps,
            )
            terminal_text, valid_terminal = _decode(terminal, family)
            terminal_correct = valid_terminal and terminal_text == str(row["answer"])
            initial_correct = tuple(trace[0]) == supervision.initial
            active_transitions = 0
            exact_transitions = 0
            trajectory_correct = initial_correct
            for index, active in enumerate(supervision.free_active):
                if not active:
                    continue
                active_transitions += 1
                exact = tuple(trace[index + 1]) == supervision.free_targets[index]
                exact_transitions += int(exact)
                trajectory_correct = trajectory_correct and exact
            packet_correct = int(actual_selection) == expected_selection
            kwargs = {
                "terminal_correct": terminal_correct,
                "packet_correct": packet_correct,
                "initial_correct": initial_correct,
                "trajectory_correct": trajectory_correct,
                "valid_terminal": valid_terminal,
                "active_transitions": active_transitions,
                "exact_transitions": exact_transitions,
            }
            _update_counts(overall, **kwargs)
            _update_counts(families[family], **kwargs)
            decoded_trace = [_decode(value, family)[0] for value in trace]
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "family": family,
                    "trace_kind": args.trace_kind,
                    "target_selection": expected_selection,
                    "selected_candidate": int(actual_selection),
                    "packet_correct": packet_correct,
                    "terminal_prediction": terminal_text,
                    "terminal_target": str(row["answer"]),
                    "terminal_correct": terminal_correct,
                    "valid_terminal": valid_terminal,
                    "initial_state_correct": initial_correct,
                    "full_trajectory_correct": trajectory_correct,
                    "active_transitions": active_transitions,
                    "exact_transitions": exact_transitions,
                    "state_trace": decoded_trace,
                }
            )

    frozen_final = model.frozen_crp_sha256()
    replay_final = module_state_sha256(model.replay)
    if frozen_initial != frozen_final or replay_initial != replay_final:
        raise RSM1EvalError("RSM1 evaluation changed model state")
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_loader,
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "crp_checkpoint_sha256": args.crp_checkpoint_sha256,
        "rsm_checkpoint_sha256": args.rsm_checkpoint_sha256,
        "checkpoint_update": update,
        "data": str(args.data.resolve()),
        "data_sha256": args.data_sha256,
        "packet_arm": args.packet_arm,
        "trace_kind": args.trace_kind,
        "selection_mode": args.selection_mode,
        "ablation": args.ablation,
        "batch_size": args.batch_size,
        "overall": overall,
        "families": families,
        "runtime_semantic_calls": 0,
        "frozen_crp_sha256": frozen_final,
        "replay_sha256": replay_final,
        "model_unchanged": True,
        "results": results,
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("auto", "causal"), default="causal")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--crp-checkpoint", type=Path, required=True)
    parser.add_argument("--crp-checkpoint-sha256", required=True)
    parser.add_argument("--rsm-checkpoint", type=Path, required=True)
    parser.add_argument("--rsm-checkpoint-sha256", required=True)
    parser.add_argument("--expected-update", type=int, default=1600)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet-arm", choices=("guarded", "unguarded"), required=True)
    parser.add_argument("--trace-kind", choices=("wrong", "correct"), required=True)
    parser.add_argument("--selection-mode", choices=("forced", "autonomous"), required=True)
    parser.add_argument(
        "--ablation",
        choices=("normal", "reset", "force_no_error", "shift", "packet_swap"),
        default="normal",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--state-width", type=int, default=256)
    parser.add_argument("--state-slots", type=int, default=24)
    parser.add_argument("--packet-slots", type=int, default=6)
    parser.add_argument("--max-trace-steps", type=int, default=12)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    args = parser.parse_args()
    integer_fields = (
        args.expected_update,
        args.batch_size,
        args.state_width,
        args.state_slots,
        args.packet_slots,
        args.max_trace_steps,
        args.attention_heads,
        args.ff_multiplier,
        args.max_sequence_length,
    )
    if any(value <= 0 for value in integer_fields):
        parser.error("RSM1 evaluation dimensions must be positive")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        f"[rsm1-eval] kind={report['trace_kind']} "
        f"selection={report['selection_mode']} "
        f"terminal={report['overall']['terminal_correct']}/480 "
        f"trajectory={report['overall']['full_trajectory_correct']}/480",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
