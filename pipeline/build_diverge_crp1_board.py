"""Build verifier-exact complete-trace causal-revision boards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Callable

from diverge_crp1_data import tokenize_revision_example


BOARD_SCHEMA = "shohin-diverge-crp1-board-v1"
REPORT_SCHEMA = "shohin-diverge-crp1-board-report-v1"
FAMILIES = ("scalar", "register", "symbolic")


class CRP1BoardError(RuntimeError):
    """The requested causal-revision board violates its contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(seed: int, label: str) -> str:
    return hashlib.sha256(f"{seed}\0{label}".encode()).hexdigest()


def _boxed(value: str) -> str:
    return f"Final answer: \\boxed{{{value}}}"


def _wrong_target(error_index: int, correct_steps: list[str], answer: str) -> str:
    replay = "\n".join(correct_steps[error_index:])
    suffix = f"\n{replay}" if replay else ""
    return (
        f"Error step: {error_index}\n"
        f"Correction: {correct_steps[error_index - 1]}{suffix}\n"
        f"{_boxed(answer)}"
    )


def _correct_target(answer: str) -> str:
    return f"Error step: NONE\nCorrection: No correction needed.\n{_boxed(answer)}"


def _scalar_episode(rng: random.Random, depth: int, heldout: bool) -> dict[str, Any]:
    start = rng.randint(31, 89) if heldout else rng.randint(3, 29)
    operations: list[tuple[str, int]] = []
    for _ in range(depth):
        kind = rng.choice(("add", "subtract", "multiply"))
        value = rng.randint(3, 12) if heldout else rng.randint(2, 9)
        if kind == "multiply":
            value = rng.randint(2, 4)
        operations.append((kind, value))

    def apply(state: int, operation: tuple[str, int]) -> int:
        kind, value = operation
        if kind == "add":
            return state + value
        if kind == "subtract":
            return state - value
        return state * value

    def line(index: int, before: int, operation: tuple[str, int], after: int) -> str:
        kind, value = operation
        symbol = {"add": "+", "subtract": "-", "multiply": "*"}[kind]
        return f"Step {index}: {before} {symbol} {value} = {after}."

    correct_states = [start]
    correct_steps: list[str] = []
    for index, operation in enumerate(operations, start=1):
        after = apply(correct_states[-1], operation)
        correct_steps.append(line(index, correct_states[-1], operation, after))
        correct_states.append(after)
    error_index = rng.randint(1, depth - 2)
    wrong_states = [start]
    wrong_steps: list[str] = []
    delta = rng.choice((-7, -5, -3, 3, 5, 7))
    for index, operation in enumerate(operations, start=1):
        after = apply(wrong_states[-1], operation)
        if index == error_index:
            after += delta
        wrong_steps.append(line(index, wrong_states[-1], operation, after))
        wrong_states.append(after)
    answer = str(correct_states[-1])
    wrong_answer = str(wrong_states[-1])
    if answer == wrong_answer:
        raise CRP1BoardError("scalar mutation did not change the answer")
    operation_text = "; ".join(f"{kind} {value}" for kind, value in operations)
    if heldout:
        problem = (
            f"Initialize scalar x at {start}. Execute, in order: {operation_text}. "
            "Report the terminal value of x."
        )
    else:
        problem = rng.choice(
            (
                f"Start with x={start}. Apply these operations in order: "
                f"{operation_text}. What is the final x?",
                f"A scalar register begins at {start}. Its program is: "
                f"{operation_text}. Give the final register value.",
            )
        )
    return {
        "family": "scalar",
        "problem": problem,
        "program": [[kind, value] for kind, value in operations],
        "initial_state": start,
        "correct_steps": correct_steps,
        "wrong_steps": wrong_steps,
        "answer": answer,
        "wrong_answer": wrong_answer,
        "error_index": error_index,
    }


def _register_episode(rng: random.Random, depth: int, heldout: bool) -> dict[str, Any]:
    initial = (
        rng.randint(13, 35) if heldout else rng.randint(2, 14),
        rng.randint(11, 31) if heldout else rng.randint(2, 14),
    )
    kinds = ("A+=B", "B-=A", "swap", "A*=2", "B+=A")
    operations = [rng.choice(kinds) for _ in range(depth)]

    def apply(state: tuple[int, int], operation: str) -> tuple[int, int]:
        a, b = state
        if operation == "A+=B":
            return a + b, b
        if operation == "B-=A":
            return a, b - a
        if operation == "swap":
            return b, a
        if operation == "A*=2":
            return 2 * a, b
        return a, b + a

    descriptions = {
        "A+=B": "add B to A",
        "B-=A": "subtract A from B",
        "swap": "swap A and B",
        "A*=2": "double A",
        "B+=A": "add A to B",
    }

    def line(
        index: int,
        before: tuple[int, int],
        operation: str,
        after: tuple[int, int],
    ) -> str:
        return (
            f"Step {index}: {descriptions[operation]}: "
            f"(A={before[0]}, B={before[1]}) -> (A={after[0]}, B={after[1]})."
        )

    correct_states = [initial]
    correct_steps: list[str] = []
    for index, operation in enumerate(operations, start=1):
        after = apply(correct_states[-1], operation)
        correct_steps.append(line(index, correct_states[-1], operation, after))
        correct_states.append(after)
    error_index = rng.randint(1, depth - 2)
    delta = rng.choice((-5, -3, 3, 5))
    wrong_states = [initial]
    wrong_steps: list[str] = []
    for index, operation in enumerate(operations, start=1):
        after = apply(wrong_states[-1], operation)
        if index == error_index:
            after = (after[0] + delta, after[1])
        wrong_steps.append(line(index, wrong_states[-1], operation, after))
        wrong_states.append(after)
    answer = f"{correct_states[-1][0]},{correct_states[-1][1]}"
    wrong_answer = f"{wrong_states[-1][0]},{wrong_states[-1][1]}"
    if answer == wrong_answer:
        raise CRP1BoardError("register mutation did not change the answer")
    operation_text = "; ".join(descriptions[value] for value in operations)
    if heldout:
        problem = (
            f"Initialize the ordered register pair as A={initial[0]}, B={initial[1]}. "
            f"Execute this noncommuting program: {operation_text}. Report final A,B."
        )
    else:
        problem = rng.choice(
            (
                f"Registers start at A={initial[0]}, B={initial[1]}. In order, "
                f"{operation_text}. What is final A,B?",
                f"Run this two-register program from ({initial[0]},{initial[1]}): "
                f"{operation_text}. Return the terminal pair A,B.",
            )
        )
    return {
        "family": "register",
        "problem": problem,
        "program": operations,
        "initial_state": list(initial),
        "correct_steps": correct_steps,
        "wrong_steps": wrong_steps,
        "answer": answer,
        "wrong_answer": wrong_answer,
        "error_index": error_index,
    }


def _symbolic_episode(rng: random.Random, depth: int, heldout: bool) -> dict[str, Any]:
    alphabet = list("abcdefghijkmnpqrstuvwxyz")
    width = rng.randint(7, 9) if heldout else rng.randint(5, 7)
    rng.shuffle(alphabet)
    initial = "".join(alphabet[:width])
    operations: list[tuple[str, int, int]] = []
    for _ in range(depth):
        kind = rng.choice(("reverse", "rotate", "swap"))
        if kind == "reverse":
            operations.append((kind, 0, 0))
        elif kind == "rotate":
            operations.append((kind, rng.randint(1, width - 1), 0))
        else:
            left, right = rng.sample(range(width), 2)
            operations.append((kind, left + 1, right + 1))

    def apply(state: str, operation: tuple[str, int, int]) -> str:
        kind, left, right = operation
        if kind == "reverse":
            return state[::-1]
        if kind == "rotate":
            return state[left:] + state[:left]
        values = list(state)
        values[left - 1], values[right - 1] = values[right - 1], values[left - 1]
        return "".join(values)

    def description(operation: tuple[str, int, int]) -> str:
        kind, left, right = operation
        if kind == "reverse":
            return "reverse the string"
        if kind == "rotate":
            return f"rotate left by {left}"
        return f"swap positions {left} and {right}"

    def line(index: int, before: str, operation: tuple[str, int, int], after: str) -> str:
        return f"Step {index}: {description(operation)}: {before} -> {after}."

    correct_states = [initial]
    correct_steps: list[str] = []
    for index, operation in enumerate(operations, start=1):
        after = apply(correct_states[-1], operation)
        correct_steps.append(line(index, correct_states[-1], operation, after))
        correct_states.append(after)
    error_index = rng.randint(1, depth - 2)
    wrong_states = [initial]
    wrong_steps: list[str] = []
    for index, operation in enumerate(operations, start=1):
        after = apply(wrong_states[-1], operation)
        if index == error_index:
            values = list(after)
            left, right = rng.sample(range(width), 2)
            values[left], values[right] = values[right], values[left]
            after = "".join(values)
        wrong_steps.append(line(index, wrong_states[-1], operation, after))
        wrong_states.append(after)
    answer = correct_states[-1]
    wrong_answer = wrong_states[-1]
    if answer == wrong_answer:
        raise CRP1BoardError("symbolic mutation did not change the answer")
    operation_text = "; ".join(description(value) for value in operations)
    if heldout:
        problem = (
            f"Initialize symbol tape T as {initial}. Execute sequentially: "
            f"{operation_text}. Report terminal T."
        )
    else:
        problem = rng.choice(
            (
                f"Start with the string {initial}. Apply in order: {operation_text}. "
                "What string remains?",
                f"A symbol tape begins as {initial}. Its program is: {operation_text}. "
                "Give the final tape.",
            )
        )
    return {
        "family": "symbolic",
        "problem": problem,
        "program": [list(value) for value in operations],
        "initial_state": initial,
        "correct_steps": correct_steps,
        "wrong_steps": wrong_steps,
        "answer": answer,
        "wrong_answer": wrong_answer,
        "error_index": error_index,
    }


BUILDERS: dict[str, Callable[[random.Random, int, bool], dict[str, Any]]] = {
    "scalar": _scalar_episode,
    "register": _register_episode,
    "symbolic": _symbolic_episode,
}


def generate_episode(seed: int, family: str, depth: int, heldout: bool) -> dict[str, Any]:
    if family not in BUILDERS or depth < 4:
        raise CRP1BoardError("episode family or depth differs")
    row = BUILDERS[family](random.Random(seed), depth, heldout)
    identity = hashlib.sha256(
        json.dumps(
            {
                "family": family,
                "problem": row["problem"],
                "correct_steps": row["correct_steps"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    row.update(
        {
            "identity_sha256": identity,
            "depth": depth,
            "heldout": heldout,
            "candidate_count": depth + 1,
            "correct_target": _correct_target(str(row["answer"])),
            "wrong_target": _wrong_target(
                int(row["error_index"]), row["correct_steps"], str(row["answer"])
            ),
        }
    )
    return row


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise CRP1BoardError(f"refusing to replace board: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise CRP1BoardError(f"refusing to replace report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _admit(tokenizer: Any, row: dict[str, Any], args: argparse.Namespace) -> bool:
    wrong = tokenize_revision_example(
        tokenizer,
        row["problem"],
        row["wrong_steps"],
        _boxed(row["wrong_answer"]),
        row["wrong_target"],
        max_sequence_length=args.max_sequence_length,
        workspace_slots=args.workspace_slots,
    )
    correct = tokenize_revision_example(
        tokenizer,
        row["problem"],
        row["correct_steps"],
        _boxed(row["answer"]),
        row["correct_target"],
        max_sequence_length=args.max_sequence_length,
        workspace_slots=args.workspace_slots,
    )
    if wrong is None or correct is None:
        return False
    row["wrong_positions"] = (
        len(wrong.prompt_ids) + args.workspace_slots + len(wrong.response_ids)
    )
    row["correct_positions"] = (
        len(correct.prompt_ids) + args.workspace_slots + len(correct.response_ids)
    )
    row["wrong_target_tokens"] = len(wrong.response_ids)
    row["correct_target_tokens"] = len(correct.response_ids)
    return True


def build(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        raise CRP1BoardError("CRP1 requires exact fast-tokenizer offsets")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    split_specs = {
        "train": (args.train_per_family, False, args.train_min_depth, args.train_max_depth),
        "development": (
            args.development_per_family,
            False,
            args.train_min_depth,
            args.train_max_depth,
        ),
        "evaluation": (
            args.evaluation_per_family,
            True,
            args.eval_min_depth,
            args.eval_max_depth,
        ),
    }
    selected: dict[str, list[dict[str, Any]]] = {}
    rejected_length = 0
    seen: set[str] = set()
    for split, (count, heldout, min_depth, max_depth) in split_specs.items():
        rows: list[dict[str, Any]] = []
        for family in FAMILIES:
            offset = 0
            while sum(row["family"] == family for row in rows) < count:
                if offset > count * 40:
                    raise CRP1BoardError(f"could not admit enough {split}/{family} rows")
                episode_seed = int(_rank(args.seed, f"{split}:{family}:{offset}")[:16], 16)
                depth_rng = random.Random(episode_seed ^ args.seed)
                depth = depth_rng.randint(min_depth, max_depth)
                row = generate_episode(episode_seed, family, depth, heldout)
                offset += 1
                if row["identity_sha256"] in seen:
                    continue
                row["schema"] = BOARD_SCHEMA
                row["split"] = split
                if not _admit(tokenizer, row, args):
                    rejected_length += 1
                    continue
                seen.add(row["identity_sha256"])
                rows.append(row)
        rows.sort(key=lambda row: _rank(args.seed, f"order:{row['identity_sha256']}"))
        selected[split] = rows

    outputs = {
        "train": args.train_output,
        "development": args.development_output,
        "evaluation": args.evaluation_output,
    }
    for split, path in outputs.items():
        _atomic_jsonl(path, selected[split])
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "seed": args.seed,
        "families": list(FAMILIES),
        "split_rows": {key: len(value) for key, value in selected.items()},
        "split_family_counts": {
            split: {
                family: sum(row["family"] == family for row in rows)
                for family in FAMILIES
            }
            for split, rows in selected.items()
        },
        "train_depths": [args.train_min_depth, args.train_max_depth],
        "evaluation_depths": [args.eval_min_depth, args.eval_max_depth],
        "evaluation_heldout_renderer_and_value_band": True,
        "identity_overlap": 0,
        "rejected_length": rejected_length,
        "complete_explicit_wrong_traces": True,
        "one_certified_first_error": True,
        "wrong_suffix_replayed_from_error": True,
        "correct_world_in_candidate_support": True,
        "supervisor_program_hidden_from_model_input": True,
        "max_sequence_length": args.max_sequence_length,
        "workspace_slots": args.workspace_slots,
        "max_positions": max(
            max(row["wrong_positions"], row["correct_positions"])
            for rows in selected.values()
            for row in rows
        ),
        "outputs": {
            split: {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
            }
            for split, path in outputs.items()
        },
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-per-family", type=int, default=1600)
    parser.add_argument("--development-per-family", type=int, default=160)
    parser.add_argument("--evaluation-per-family", type=int, default=160)
    parser.add_argument("--train-min-depth", type=int, default=4)
    parser.add_argument("--train-max-depth", type=int, default=6)
    parser.add_argument("--eval-min-depth", type=int, default=7)
    parser.add_argument("--eval-max-depth", type=int, default=9)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--workspace-slots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026080604)
    args = parser.parse_args()
    if min(
        args.train_per_family,
        args.development_per_family,
        args.evaluation_per_family,
        args.train_min_depth,
        args.max_sequence_length,
        args.workspace_slots,
    ) <= 0:
        parser.error("CRP1 board dimensions must be positive")
    if not 4 <= args.train_min_depth <= args.train_max_depth < args.eval_min_depth <= args.eval_max_depth:
        parser.error("CRP1 depth bands must be disjoint and ordered")
    return args


def main() -> int:
    report = build(parse_args())
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
