#!/usr/bin/env python3
"""Generate decontaminated function-completion tasks from typed computation graphs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable


SCHEMA = "shohin-verified-function-graph-v1"
WORD = re.compile(r"\w+")


class FunctionGraphCorpusError(RuntimeError):
    """A generated function corpus violates its admission contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _grams(value: str, width: int) -> set[str]:
    words = WORD.findall(value.casefold())
    if not words:
        return set()
    if len(words) < width:
        return {" ".join(words)}
    return {
        " ".join(words[index : index + width])
        for index in range(len(words) - width + 1)
    }


def _question(row: dict[str, Any]) -> str:
    for key in ("question", "problem", "prompt", "text", "input"):
        if row.get(key):
            return str(row[key])
    return ""


def evaluation_grams(
    paths: list[Path], width: int
) -> tuple[set[str], list[dict[str, Any]]]:
    grams: set[str] = set()
    receipts: list[dict[str, Any]] = []
    for path in paths:
        payload = path.read_bytes()
        rows = 0
        for raw in payload.splitlines():
            if not raw:
                continue
            row = json.loads(raw)
            grams.update(_grams(_question(row), width))
            rows += 1
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_bytes(payload),
                "rows": rows,
            }
        )
    return grams, receipts


def _function_name(family: str, identity: int) -> str:
    suffix = hashlib.sha256(f"{family}\0{identity}".encode()).hexdigest()[:10]
    return f"{family}_{suffix}"


def _program_passes(code: str, tests: list[str], timeout_seconds: float) -> bool:
    source = code + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory(prefix="shohin-function-graph-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def _prompt(name: str, signature: str, rules: list[str], examples: list[str]) -> str:
    numbered = "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, 1))
    shown = "\n".join(examples)
    return (
        f"Implement the Python function `{name}`.\n"
        f"Signature: `{signature}`\n"
        "Apply these rules in order:\n"
        f"{numbered}\n"
        "Examples:\n"
        f"{shown}\n"
        "Return only the complete Python function."
    )


def _list_task(rng: random.Random, identity: int) -> dict[str, Any]:
    name = _function_name("number_pipeline", identity)
    modulus = rng.randint(2, 7)
    residue = rng.randrange(modulus)
    scale = rng.choice([-3, -2, 2, 3, 4])
    offset = rng.randint(-8, 8)
    reverse = bool(rng.getrandbits(1))
    deduplicate = bool(rng.getrandbits(1))
    result_kind = rng.choice(("values", "sum", "count"))

    def solve(values: list[int]) -> list[int] | int:
        current = [
            value * scale + offset for value in values if value % modulus == residue
        ]
        if deduplicate:
            current = list(dict.fromkeys(current))
        current.sort(reverse=reverse)
        if result_kind == "sum":
            return sum(current)
        if result_kind == "count":
            return len(current)
        return current

    return_annotation = "int" if result_kind != "values" else "list[int]"
    code = [
        f"def {name}(values: list[int]) -> {return_annotation}:",
        "    current = []",
        "    for value in values:",
        f"        if value % {modulus} == {residue}:",
        f"            current.append(value * {scale} + ({offset}))",
    ]
    if deduplicate:
        code.extend(
            [
                "    seen = set()",
                "    current = [value for value in current if not (value in seen or seen.add(value))]",
            ]
        )
    code.append(f"    current.sort(reverse={reverse})")
    if result_kind == "sum":
        code.append("    return sum(current)")
    elif result_kind == "count":
        code.append("    return len(current)")
    else:
        code.append("    return current")
    inputs = [
        [rng.randint(-20, 30) for _ in range(rng.randint(0, 14))] for _ in range(10)
    ]
    rules = [
        f"Keep only integers whose remainder modulo {modulus} is {residue}.",
        f"Replace every retained integer x by x * {scale} + ({offset}).",
    ]
    if deduplicate:
        rules.append("Remove duplicates while preserving their first occurrence.")
    rules.append(
        f"Sort the retained values in {'descending' if reverse else 'ascending'} order."
    )
    rules.append(
        {
            "values": "Return the resulting list.",
            "sum": "Return its sum (zero when empty).",
            "count": "Return its length.",
        }[result_kind]
    )
    signature = f"def {name}(values: list[int]) -> {return_annotation}:"
    return {
        "family": "list_pipeline",
        "name": name,
        "question": _prompt(
            name,
            signature,
            rules,
            [f"{name}({value!r}) == {solve(value)!r}" for value in inputs[:2]],
        ),
        "response": "\n".join(code),
        "tests": [f"assert {name}({value!r}) == {solve(value)!r}" for value in inputs],
        "graph": {
            "modulus": modulus,
            "residue": residue,
            "scale": scale,
            "offset": offset,
            "deduplicate": deduplicate,
            "reverse": reverse,
            "result": result_kind,
        },
    }


def _string_task(rng: random.Random, identity: int) -> dict[str, Any]:
    name = _function_name("text_pipeline", identity)
    keep = rng.choice(("letters", "alphanumeric", "digits", "nonspace"))
    case = rng.choice(("lower", "upper", "unchanged"))
    reverse = bool(rng.getrandbits(1))
    collapse = bool(rng.getrandbits(1))
    result_kind = rng.choice(("text", "length", "vowels"))

    def selected(character: str) -> bool:
        if keep == "nonspace":
            return not character.isspace()
        return {
            "letters": character.isalpha(),
            "alphanumeric": character.isalnum(),
            "digits": character.isdigit(),
        }[keep]

    def solve(text: str) -> str | int:
        current = "".join(character for character in text if selected(character))
        if case == "lower":
            current = current.lower()
        elif case == "upper":
            current = current.upper()
        if collapse:
            current = "".join(
                character
                for index, character in enumerate(current)
                if index == 0 or character != current[index - 1]
            )
        if reverse:
            current = current[::-1]
        if result_kind == "length":
            return len(current)
        if result_kind == "vowels":
            return sum(character.casefold() in "aeiou" for character in current)
        return current

    predicate = {
        "letters": "character.isalpha()",
        "alphanumeric": "character.isalnum()",
        "digits": "character.isdigit()",
        "nonspace": "not character.isspace()",
    }[keep]
    annotation = "str" if result_kind == "text" else "int"
    code = [
        f"def {name}(text: str) -> {annotation}:",
        f"    current = ''.join(character for character in text if {predicate})",
    ]
    if case != "unchanged":
        code.append(f"    current = current.{case}()")
    if collapse:
        code.extend(
            [
                "    collapsed = []",
                "    for character in current:",
                "        if not collapsed or collapsed[-1] != character:",
                "            collapsed.append(character)",
                "    current = ''.join(collapsed)",
            ]
        )
    if reverse:
        code.append("    current = current[::-1]")
    if result_kind == "length":
        code.append("    return len(current)")
    elif result_kind == "vowels":
        code.append(
            "    return sum(character.casefold() in 'aeiou' for character in current)"
        )
    else:
        code.append("    return current")
    alphabet = "aAbBcCxyzXYZ001122  -_!?"
    inputs = [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
        for _ in range(10)
    ]
    rules = [f"Keep only {keep} characters."]
    if case != "unchanged":
        rules.append(f"Convert the retained text to {case} case.")
    if collapse:
        rules.append(
            "Collapse every run of identical adjacent characters to one character."
        )
    if reverse:
        rules.append("Reverse the resulting character sequence.")
    rules.append(
        {
            "text": "Return the text.",
            "length": "Return its length.",
            "vowels": "Return its vowel count.",
        }[result_kind]
    )
    signature = f"def {name}(text: str) -> {annotation}:"
    return {
        "family": "string_pipeline",
        "name": name,
        "question": _prompt(
            name,
            signature,
            rules,
            [f"{name}({value!r}) == {solve(value)!r}" for value in inputs[:2]],
        ),
        "response": "\n".join(code),
        "tests": [f"assert {name}({value!r}) == {solve(value)!r}" for value in inputs],
        "graph": {
            "keep": keep,
            "case": case,
            "collapse": collapse,
            "reverse": reverse,
            "result": result_kind,
        },
    }


def _number_task(rng: random.Random, identity: int) -> dict[str, Any]:
    name = _function_name("integer_fold", identity)
    family = rng.choice(("digits", "divisors", "collatz"))
    parity = rng.choice((0, 1))
    result_kind = rng.choice(("sum", "count"))

    def solve(n: int) -> int:
        value = abs(n)
        if family == "digits":
            items = [
                int(character)
                for character in str(value)
                if int(character) % 2 == parity
            ]
        elif family == "divisors":
            items = [
                divisor
                for divisor in range(1, value + 1)
                if value % divisor == 0 and divisor % 2 == parity
            ]
        else:
            items = []
            current = max(1, value)
            while True:
                if current % 2 == parity:
                    items.append(current)
                if current == 1:
                    break
                current = current // 2 if current % 2 == 0 else 3 * current + 1
        return sum(items) if result_kind == "sum" else len(items)

    code = [f"def {name}(n: int) -> int:", "    value = abs(n)"]
    if family == "digits":
        code.append(
            f"    items = [int(ch) for ch in str(value) if int(ch) % 2 == {parity}]"
        )
    elif family == "divisors":
        code.append(
            f"    items = [d for d in range(1, value + 1) if value % d == 0 and d % 2 == {parity}]"
        )
    else:
        code.extend(
            [
                "    items = []",
                "    current = max(1, value)",
                "    while True:",
                f"        if current % 2 == {parity}:",
                "            items.append(current)",
                "        if current == 1:",
                "            break",
                "        current = current // 2 if current % 2 == 0 else 3 * current + 1",
            ]
        )
    code.append(
        "    return sum(items)" if result_kind == "sum" else "    return len(items)"
    )
    inputs = [
        rng.randint(-5000, 5000) if family != "collatz" else rng.randint(1, 300)
        for _ in range(10)
    ]
    noun = {
        "digits": "base-10 digits",
        "divisors": "positive divisors",
        "collatz": "values in the Collatz sequence including the start and 1",
    }[family]
    rules = [
        "Use the absolute value of n.",
        f"Collect the {noun} whose parity is {'even' if parity == 0 else 'odd'}.",
        f"Return the {'sum' if result_kind == 'sum' else 'count'} of the collected values.",
    ]
    signature = f"def {name}(n: int) -> int:"
    return {
        "family": "number_theory",
        "name": name,
        "question": _prompt(
            name,
            signature,
            rules,
            [f"{name}({value!r}) == {solve(value)!r}" for value in inputs[:2]],
        ),
        "response": "\n".join(code),
        "tests": [f"assert {name}({value!r}) == {solve(value)!r}" for value in inputs],
        "graph": {"source": family, "parity": parity, "result": result_kind},
    }


def _record_task(rng: random.Random, identity: int) -> dict[str, Any]:
    name = _function_name("record_pipeline", identity)
    threshold = rng.randint(-5, 20)
    reverse = bool(rng.getrandbits(1))
    result_kind = rng.choice(("names", "total", "best"))

    def solve(records: list[tuple[str, int]]) -> list[str] | int | str | None:
        kept = [(label, score) for label, score in records if score >= threshold]
        kept.sort(key=lambda item: (item[1], item[0]), reverse=reverse)
        if result_kind == "names":
            return [label for label, _score in kept]
        if result_kind == "total":
            return sum(score for _label, score in kept)
        return kept[0][0] if kept else None

    annotation = {"names": "list[str]", "total": "int", "best": "str | None"}[
        result_kind
    ]
    code = [
        f"def {name}(records: list[tuple[str, int]]) -> {annotation}:",
        f"    kept = [(label, score) for label, score in records if score >= {threshold}]",
        f"    kept.sort(key=lambda item: (item[1], item[0]), reverse={reverse})",
    ]
    if result_kind == "names":
        code.append("    return [label for label, _score in kept]")
    elif result_kind == "total":
        code.append("    return sum(score for _label, score in kept)")
    else:
        code.append("    return kept[0][0] if kept else None")
    labels = ["amy", "bo", "cy", "dia", "eli", "fox"]
    inputs = [
        [(rng.choice(labels), rng.randint(-10, 30)) for _ in range(rng.randint(0, 10))]
        for _ in range(10)
    ]
    rules = [
        f"Keep records whose integer score is at least {threshold}.",
        f"Sort retained records by (score, name) in {'descending' if reverse else 'ascending'} order.",
        {
            "names": "Return the names in that order.",
            "total": "Return the sum of retained scores.",
            "best": "Return the first retained name, or None when none remain.",
        }[result_kind],
    ]
    signature = f"def {name}(records: list[tuple[str, int]]) -> {annotation}:"
    return {
        "family": "record_pipeline",
        "name": name,
        "question": _prompt(
            name,
            signature,
            rules,
            [f"{name}({value!r}) == {solve(value)!r}" for value in inputs[:2]],
        ),
        "response": "\n".join(code),
        "tests": [f"assert {name}({value!r}) == {solve(value)!r}" for value in inputs],
        "graph": {"threshold": threshold, "reverse": reverse, "result": result_kind},
    }


GENERATORS: tuple[Callable[[random.Random, int], dict[str, Any]], ...] = (
    _list_task,
    _string_task,
    _number_task,
    _record_task,
)


def generate_rows(
    *,
    count: int,
    seed: int,
    shard_index: int,
    shard_count: int,
    blocked_grams: set[str],
    ngram_width: int,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if count <= 0 or shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise FunctionGraphCorpusError("generation shape differs")
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    seen: set[str] = set()
    local = 0
    while len(rows) < count:
        global_identity = shard_index + local * shard_count
        local += 1
        rng = random.Random(seed + global_identity * 1000003)
        generator = GENERATORS[global_identity % len(GENERATORS)]
        task = generator(rng, global_identity)
        counters["generated"] += 1
        normalized = re.sub(r"\s+", " ", task["question"]).strip().casefold()
        if normalized in seen:
            counters["duplicate_question"] += 1
            continue
        if blocked_grams.intersection(_grams(task["question"], ngram_width)):
            counters["eval_ngram_overlap"] += 1
            continue
        if not _program_passes(task["response"], task["tests"], timeout_seconds):
            counters["execution_failure"] += 1
            continue
        seen.add(normalized)
        verification_sha = _sha256_bytes(
            (task["response"] + "\n" + "\n".join(task["tests"])).encode()
        )
        row = {
            **task,
            "schema": SCHEMA,
            "source": "shohin_typed_function_graph",
            "training_group": "code",
            "verification": "generated_reference_passes_randomized_tests",
            "verification_sha256": verification_sha,
            "seed": seed,
            "global_identity": global_identity,
            "split": "confirmation"
            if int(verification_sha[:8], 16) % 20 == 0
            else "train",
        }
        rows.append(row)
        counters["kept"] += 1
        counters[f"family_{row['family']}"] += 1
        counters[f"split_{row['split']}"] += 1
    return rows, counters


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise FunctionGraphCorpusError(f"refusing existing output: {path}")
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
        raise FunctionGraphCorpusError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--eval", type=Path, action="append", default=[])
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--ngram-width", type=int, default=13)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    args = parser.parse_args()
    blocked, receipts = evaluation_grams(args.eval, args.ngram_width)
    rows, counters = generate_rows(
        count=args.count,
        seed=args.seed,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        blocked_grams=blocked,
        ngram_width=args.ngram_width,
        timeout_seconds=args.timeout_seconds,
    )
    output_sha = _atomic_lines(args.output, rows)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "output": str(args.output.resolve()),
        "output_sha256": output_sha,
        "rows": len(rows),
        "seed": args.seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "ngram_width": args.ngram_width,
        "evaluation_sources": receipts,
        "counters": dict(sorted(counters.items())),
    }
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
