#!/usr/bin/env python3
"""Build decontaminated MBPP bug-to-fix trajectories with executed failures."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = "shohin-verified-code-repair-curriculum-v1"
SYSTEM_TASK = (
    "Write Python code that solves the task and passes every test. Return only "
    "executable Python code, without Markdown fences."
)


class VerifiedCodeRepairError(RuntimeError):
    """The requested repair curriculum violates its data contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_bytes().splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifiedCodeRepairError(f"malformed JSONL: {path}") from exc


def _program_result(program: str, timeout_seconds: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="shohin-repair-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(program)
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=directory,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                env={**os.environ, "PYTHONHASHSEED": "0"},
            )
            return {
                "passed": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "returncode": None,
                "stdout": str(exc.stdout or "")[-2000:],
                "stderr": "execution timed out",
            }


def _test_program(code: str, row: dict[str, Any]) -> str:
    setup = str(row.get("test_setup_code") or "")
    tests = "\n".join(str(value) for value in row.get("test_list") or ())
    return f"{code}\n{setup}\n{tests}\n"


class _ReplaceNode(ast.NodeTransformer):
    def __init__(self, target: ast.AST, replacement: ast.AST):
        self.target = target
        self.replacement = replacement

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if node is self.target:
            return ast.copy_location(self.replacement, node)
        return super().generic_visit(node)


def _replacement(node: ast.AST) -> tuple[ast.AST, str] | None:
    binary = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.FloorDiv,
        ast.FloorDiv: ast.Mult,
        ast.Div: ast.Mult,
        ast.Mod: ast.Mult,
        ast.Pow: ast.Mult,
        ast.BitAnd: ast.BitOr,
        ast.BitOr: ast.BitAnd,
    }
    comparisons = {
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
        ast.In: ast.NotIn,
        ast.NotIn: ast.In,
        ast.Is: ast.IsNot,
        ast.IsNot: ast.Is,
    }
    if isinstance(node, ast.operator) and type(node) in binary:
        replacement = binary[type(node)]()
        return replacement, f"operator:{type(node).__name__}->{type(replacement).__name__}"
    if isinstance(node, ast.cmpop) and type(node) in comparisons:
        replacement = comparisons[type(node)]()
        return replacement, f"comparison:{type(node).__name__}->{type(replacement).__name__}"
    if isinstance(node, ast.And):
        return ast.Or(), "boolean:And->Or"
    if isinstance(node, ast.Or):
        return ast.And(), "boolean:Or->And"
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return ast.Constant(not node.value), f"constant:{node.value}->{not node.value}"
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and abs(node.value) <= 10000
    ):
        delta = 1 if node.value >= 0 else -1
        return ast.Constant(node.value + delta), f"constant:{node.value}->{node.value + delta}"
    return None


def mutation_candidates(code: str, seed_material: str) -> list[tuple[str, str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise VerifiedCodeRepairError("verified anchor code does not parse") from exc
    candidates: dict[str, str] = {}
    for index, node in enumerate(ast.walk(tree)):
        replacement = _replacement(node)
        if replacement is None:
            continue
        replacement_node, label = replacement
        mutated = ast.parse(code)
        target = list(ast.walk(mutated))[index]
        mutated = _ReplaceNode(target, replacement_node).visit(mutated)
        ast.fix_missing_locations(mutated)
        rendered = ast.unparse(mutated).strip()
        if rendered != code.strip():
            candidates.setdefault(rendered, f"{label}@{index}")
    return sorted(
        candidates.items(),
        key=lambda item: hashlib.sha256(
            f"{seed_material}\0{item[0]}".encode()
        ).digest(),
    )


def _diagnostic(result: dict[str, Any]) -> str:
    fields = []
    for key in ("returncode", "stdout", "stderr"):
        value = result.get(key)
        if value not in (None, ""):
            fields.append(f"{key}: {value}")
    return ("\n".join(fields) or "The shown tests did not pass.")[:2000]


def build_source_repairs(
    raw: dict[str, Any],
    anchor: dict[str, Any],
    *,
    mutations_per_source: int,
    max_candidates: int,
    timeout_seconds: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    question = str(raw.get("text") or "")
    code = str(raw.get("code") or "").strip()
    tests = [str(value) for value in raw.get("test_list") or ()]
    if (
        question != str(anchor.get("question") or "")
        or code != str(anchor.get("response") or "").strip()
        or not tests
    ):
        raise VerifiedCodeRepairError("raw MBPP row differs from admitted anchor")
    original_result = _program_result(_test_program(code, raw), timeout_seconds)
    if not original_result["passed"]:
        raise VerifiedCodeRepairError("admitted anchor no longer passes its tests")
    kept = []
    candidates = mutation_candidates(code, f"{seed}\0{question}")
    attempted = 0
    for mutant, label in candidates[:max_candidates]:
        attempted += 1
        result = _program_result(_test_program(mutant, raw), timeout_seconds)
        if result["passed"]:
            continue
        repair_instruction = (
            "Repair the previous Python solution. Return only the complete corrected "
            "executable Python code, without Markdown fences or explanation.\n\n"
            f"Original task:\n{question}\n\nPrevious solution:\n{mutant}\n\n"
            "Observed result from executing only the public tests shown below:\n"
            f"{_diagnostic(result)}\n\n"
            "Correct every defect while preserving the requested function interface."
        )
        prompt = (
            f"{SYSTEM_TASK}\n\nTask:\n{repair_instruction}\n\nTests:\n"
            + "\n".join(tests)
        )
        identity = hashlib.sha256(
            f"{question}\0{mutant}\0{label}".encode()
        ).hexdigest()
        kept.append(
            {
                "question": prompt,
                "response": code,
                "source": "mbpp_verified_mutation_repair",
                "source_split": str(anchor.get("source") or ""),
                "task_id": int(raw["task_id"]),
                "repair_identity_sha256": identity,
                "mutation": label,
                "mutant_sha256": hashlib.sha256(mutant.encode()).hexdigest(),
                "target_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "test_list_sha256": hashlib.sha256(
                    json.dumps(tests, sort_keys=True).encode()
                ).hexdigest(),
                "failure_execution": result,
            }
        )
        if len(kept) >= mutations_per_source:
            break
    return kept, {"candidates": len(candidates), "attempted": attempted}


def build_curriculum(
    raw_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    *,
    mutations_per_source: int,
    max_candidates: int,
    timeout_seconds: float,
    seed: int,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if min(mutations_per_source, max_candidates, workers) <= 0 or timeout_seconds <= 0:
        raise VerifiedCodeRepairError("repair build limits must be positive")
    anchor_by_question = {
        str(row.get("question") or ""): row for row in anchor_rows
    }
    raw_by_question = {str(row.get("text") or ""): row for row in raw_rows}
    if (
        len(anchor_by_question) != len(anchor_rows)
        or len(raw_by_question) != len(raw_rows)
        or set(anchor_by_question) != set(raw_by_question)
    ):
        raise VerifiedCodeRepairError("raw and admitted anchor identities differ")
    questions = sorted(anchor_by_question)

    def one(question: str):
        return build_source_repairs(
            raw_by_question[question],
            anchor_by_question[question],
            mutations_per_source=mutations_per_source,
            max_candidates=max_candidates,
            timeout_seconds=timeout_seconds,
            seed=seed,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        built = list(executor.map(one, questions))
    rows = [row for source_rows, _ in built for row in source_rows]
    random.Random(seed).shuffle(rows)
    if len({row["repair_identity_sha256"] for row in rows}) != len(rows):
        raise VerifiedCodeRepairError("repair identities repeat")
    sources_with_repairs = sum(bool(source_rows) for source_rows, _ in built)
    return rows, {
        "schema": SCHEMA,
        "status": "complete",
        "seed": seed,
        "anchor_rows": len(anchor_rows),
        "sources_with_repairs": sources_with_repairs,
        "sources_without_repairs": len(anchor_rows) - sources_with_repairs,
        "repair_rows": len(rows),
        "mutations_per_source": mutations_per_source,
        "max_candidates_per_source": max_candidates,
        "candidate_mutations": sum(meta["candidates"] for _, meta in built),
        "executed_mutations": sum(meta["attempted"] for _, meta in built),
        "timeout_seconds": timeout_seconds,
        "workers": workers,
    }


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VerifiedCodeRepairError(f"refusing existing output: {path}")
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise VerifiedCodeRepairError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_raw(revision: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = []
    for split in ("train", "validation"):
        rows.extend(
            dict(row)
            for row in load_dataset(
                "google-research-datasets/mbpp",
                "full",
                split=split,
                revision=revision,
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--mutations-per-source", type=int, default=6)
    parser.add_argument("--max-candidates-per-source", type=int, default=48)
    parser.add_argument("--timeout-seconds", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.anchor) != args.anchor_sha256:
        raise VerifiedCodeRepairError("anchor hash differs")
    anchors = _jsonl(args.anchor)
    admitted = {str(row["question"]): row for row in anchors}
    raw = [row for row in _load_raw(args.dataset_revision) if str(row["text"]) in admitted]
    rows, report = build_curriculum(
        raw,
        anchors,
        mutations_per_source=args.mutations_per_source,
        max_candidates=args.max_candidates_per_source,
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
        workers=args.workers,
    )
    if not rows:
        raise VerifiedCodeRepairError("repair curriculum is empty")
    report.update(
        {
            "anchor": str(args.anchor.resolve()),
            "anchor_sha256": _sha256(args.anchor),
            "dataset": "google-research-datasets/mbpp:full",
            "dataset_revision": args.dataset_revision,
            "output": str(args.output.resolve()),
        }
    )
    report["output_sha256"] = _atomic_lines(args.output, rows)
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
