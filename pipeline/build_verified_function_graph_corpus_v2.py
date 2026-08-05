#!/usr/bin/env python3
"""Generate execution-verified functions aligned to measured code failures."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Callable

from build_verified_function_graph_corpus import (
    _atomic_json,
    _atomic_lines,
    _function_name,
    _grams,
    _program_passes,
    _prompt,
    evaluation_grams,
)


SCHEMA = "shohin-verified-function-graph-v2"


class FunctionGraphV2Error(RuntimeError):
    """A failure-aligned function graph violates its admission contract."""


def _index_rewrite(rng: random.Random, identity: int) -> dict[str, Any]:
    name = _function_name("index_rewrite", identity)
    modulus = rng.randint(2, 7)
    residue = rng.randrange(modulus)
    scale = rng.choice((-2, -1, 1, 2, 3))
    offset = rng.randint(-5, 5)
    reverse = bool(rng.getrandbits(1))

    def solve(values: list[int]) -> list[int]:
        output = values.copy()
        positions = [
            index for index in range(len(values)) if index % modulus == residue
        ]
        replacements = sorted(
            (values[index] * scale + offset for index in positions), reverse=reverse
        )
        for index, value in zip(positions, replacements, strict=True):
            output[index] = value
        return output

    code = "\n".join(
        (
            f"def {name}(values: list[int]) -> list[int]:",
            "    output = values.copy()",
            f"    positions = [i for i in range(len(values)) if i % {modulus} == {residue}]",
            f"    replacements = sorted((values[i] * {scale} + ({offset}) for i in positions), reverse={reverse})",
            "    for i, value in zip(positions, replacements):",
            "        output[i] = value",
            "    return output",
        )
    )
    inputs = [
        [rng.randint(-20, 30) for _ in range(rng.randint(0, 18))] for _ in range(10)
    ]
    signature = f"def {name}(values: list[int]) -> list[int]:"
    rules = [
        f"Select positions i whose remainder i modulo {modulus} is {residue}.",
        f"Transform each selected value x to x * {scale} + ({offset}).",
        f"Sort only the transformed selected values in {'descending' if reverse else 'ascending'} order.",
        "Write them back to the selected positions; every other position stays unchanged.",
        "Return the resulting list without modifying the input list.",
    ]
    return {
        "family": "index_rewrite",
        "name": name,
        "question": _prompt(
            name,
            signature,
            rules,
            [f"{name}({x!r}) == {solve(x)!r}" for x in inputs[:2]],
        ),
        "response": code,
        "tests": [f"assert {name}({x!r}) == {solve(x)!r}" for x in inputs],
        "graph": {
            "modulus": modulus,
            "residue": residue,
            "scale": scale,
            "offset": offset,
            "reverse": reverse,
        },
    }


def _frequency_filter(rng: random.Random, identity: int) -> dict[str, Any]:
    name = _function_name("frequency_filter", identity)
    minimum = rng.randint(1, 3)
    maximum = minimum + rng.randint(0, 2)
    unique = bool(rng.getrandbits(1))
    result = rng.choice(("values", "count", "sum"))

    def solve(values: list[int]) -> list[int] | int:
        counts = {value: values.count(value) for value in values}
        kept = [value for value in values if minimum <= counts[value] <= maximum]
        if unique:
            kept = list(dict.fromkeys(kept))
        if result == "count":
            return len(kept)
        if result == "sum":
            return sum(kept)
        return kept

    annotation = "list[int]" if result == "values" else "int"
    lines = [
        f"def {name}(values: list[int]) -> {annotation}:",
        "    counts = {value: values.count(value) for value in values}",
        f"    kept = [value for value in values if {minimum} <= counts[value] <= {maximum}]",
    ]
    if unique:
        lines.extend(
            (
                "    seen = set()",
                "    kept = [x for x in kept if not (x in seen or seen.add(x))]",
            )
        )
    lines.append(
        {
            "values": "    return kept",
            "count": "    return len(kept)",
            "sum": "    return sum(kept)",
        }[result]
    )
    inputs = [
        [rng.randint(-4, 7) for _ in range(rng.randint(0, 20))] for _ in range(10)
    ]
    signature = f"def {name}(values: list[int]) -> {annotation}:"
    rules = [
        f"Keep values whose total occurrence count in the original list is between {minimum} and {maximum}, inclusive.",
        "Preserve original order and occurrences.",
    ]
    if unique:
        rules.append("Then keep only the first retained occurrence of each value.")
    rules.append(
        {
            "values": "Return the retained list.",
            "count": "Return the retained length.",
            "sum": "Return the retained sum.",
        }[result]
    )
    return {
        "family": "frequency_filter",
        "name": name,
        "question": _prompt(
            name,
            signature,
            rules,
            [f"{name}({x!r}) == {solve(x)!r}" for x in inputs[:2]],
        ),
        "response": "\n".join(lines),
        "tests": [f"assert {name}({x!r}) == {solve(x)!r}" for x in inputs],
        "graph": {
            "minimum": minimum,
            "maximum": maximum,
            "unique": unique,
            "result": result,
        },
    }


def _nested_support(rng: random.Random, identity: int) -> dict[str, Any]:
    name = _function_name("nested_support", identity)
    minimum_rows = rng.randint(1, 4)
    scale = rng.choice((-2, -1, 1, 2, 3))
    offset = rng.randint(-5, 5)
    reverse = bool(rng.getrandbits(1))
    result = rng.choice(("values", "count", "sum"))

    def solve(rows: list[list[int]]) -> list[int] | int:
        universe = set().union(*(set(row) for row in rows)) if rows else set()
        kept = [
            value
            for value in universe
            if sum(value in set(row) for row in rows) >= minimum_rows
        ]
        kept = sorted((value * scale + offset for value in kept), reverse=reverse)
        if result == "count":
            return len(kept)
        if result == "sum":
            return sum(kept)
        return kept

    annotation = "list[int]" if result == "values" else "int"
    lines = [
        f"def {name}(rows: list[list[int]]) -> {annotation}:",
        "    universe = set().union(*(set(row) for row in rows)) if rows else set()",
        f"    kept = [value for value in universe if sum(value in set(row) for row in rows) >= {minimum_rows}]",
        f"    kept = sorted((value * {scale} + ({offset}) for value in kept), reverse={reverse})",
        {
            "values": "    return kept",
            "count": "    return len(kept)",
            "sum": "    return sum(kept)",
        }[result],
    ]
    inputs = [
        [
            [rng.randint(-5, 12) for _ in range(rng.randint(0, 9))]
            for _ in range(rng.randint(0, 6))
        ]
        for _ in range(10)
    ]
    signature = f"def {name}(rows: list[list[int]]) -> {annotation}:"
    rules = [
        f"Select each distinct integer that appears in at least {minimum_rows} different inner lists.",
        f"Transform each selected integer x to x * {scale} + ({offset}).",
        f"Sort the transformed values in {'descending' if reverse else 'ascending'} order.",
        {
            "values": "Return the sorted values.",
            "count": "Return their count.",
            "sum": "Return their sum.",
        }[result],
    ]
    return {
        "family": "nested_support",
        "name": name,
        "question": _prompt(
            name,
            signature,
            rules,
            [f"{name}({x!r}) == {solve(x)!r}" for x in inputs[:2]],
        ),
        "response": "\n".join(lines),
        "tests": [f"assert {name}({x!r}) == {solve(x)!r}" for x in inputs],
        "graph": {
            "minimum_rows": minimum_rows,
            "scale": scale,
            "offset": offset,
            "reverse": reverse,
            "result": result,
        },
    }


def _sentence_scan(rng: random.Random, identity: int) -> dict[str, Any]:
    name = _function_name("sentence_scan", identity)
    target = rng.choice(("I", "We", "The", "A"))
    case_sensitive = bool(rng.getrandbits(1))
    result = rng.choice(("count", "indices"))

    def solve(text: str) -> int | list[int]:
        segments = [part.strip() for part in re.split(r"[.?!]", text) if part.strip()]
        matches = []
        for index, segment in enumerate(segments):
            first = segment.split()[0] if segment.split() else ""
            equal = (
                first == target
                if case_sensitive
                else first.casefold() == target.casefold()
            )
            if equal:
                matches.append(index)
        return len(matches) if result == "count" else matches

    annotation = "int" if result == "count" else "list[int]"
    compare = (
        "first == target" if case_sensitive else "first.casefold() == target.casefold()"
    )
    lines = [
        f"def {name}(text: str) -> {annotation}:",
        "    import re",
        "    segments = [part.strip() for part in re.split(r'[.?!]', text) if part.strip()]",
        "    matches = []",
        "    for index, segment in enumerate(segments):",
        "        first = segment.split()[0] if segment.split() else ''",
        f"        target = {target!r}",
        f"        if {compare}:",
        "            matches.append(index)",
        "    return len(matches)" if result == "count" else "    return matches",
    ]
    starts = ("I", "i", "We", "we", "The", "the", "A", "a", "You")
    tails = ("agree", "will go", "saw it", "can help", "stayed home")
    inputs = []
    for _ in range(10):
        pieces = [
            f"{rng.choice(starts)} {rng.choice(tails)}"
            for _ in range(rng.randint(0, 8))
        ]
        text = "".join(
            piece + rng.choice((". ", "? ", "! ")) for piece in pieces
        ).strip()
        inputs.append(text)
    signature = f"def {name}(text: str) -> {annotation}:"
    rules = [
        "Split nonempty sentences on period, question mark, or exclamation mark.",
        f"A sentence matches when its first whitespace-delimited word {'exactly equals' if case_sensitive else 'equals ignoring case'} {target!r}.",
        "Number nonempty sentences from zero after splitting.",
        "Return the match count."
        if result == "count"
        else "Return the matching sentence indices.",
    ]
    return {
        "family": "sentence_scan",
        "name": name,
        "question": _prompt(
            name,
            signature,
            rules,
            [f"{name}({x!r}) == {solve(x)!r}" for x in inputs[:2]],
        ),
        "response": "\n".join(lines),
        "tests": [f"assert {name}({x!r}) == {solve(x)!r}" for x in inputs],
        "graph": {"target": target, "case_sensitive": case_sensitive, "result": result},
    }


def _pair_scan(rng: random.Random, identity: int) -> dict[str, Any]:
    name = _function_name("pair_scan", identity)
    mode = rng.choice(("difference", "adjacent", "palindrome"))
    parameter = rng.randint(1, 8)
    result = rng.choice(("count", "indices"))

    def solve(values: list[int]) -> int | list[int]:
        if mode == "difference":
            unique = set(values)
            found = sorted(value for value in unique if value + parameter in unique)
        elif mode == "adjacent":
            found = [
                index
                for index in range(1, len(values))
                if values[index] < values[index - 1]
            ]
        else:
            found = [
                index
                for index in range(len(values) // 2)
                if values[index] != values[-1 - index]
            ]
        return len(found) if result == "count" else found

    annotation = "int" if result == "count" else "list[int]"
    lines = [f"def {name}(values: list[int]) -> {annotation}:"]
    if mode == "difference":
        lines.extend(
            (
                "    unique = set(values)",
                f"    found = sorted(value for value in unique if value + {parameter} in unique)",
            )
        )
    elif mode == "adjacent":
        lines.append(
            "    found = [i for i in range(1, len(values)) if values[i] < values[i - 1]]"
        )
    else:
        lines.append(
            "    found = [i for i in range(len(values) // 2) if values[i] != values[-1 - i]]"
        )
    lines.append("    return len(found)" if result == "count" else "    return found")
    inputs = [
        [rng.randint(-12, 18) for _ in range(rng.randint(0, 20))] for _ in range(10)
    ]
    signature = f"def {name}(values: list[int]) -> {annotation}:"
    operation = {
        "difference": f"Find each distinct smaller value x for which x + {parameter} is also present.",
        "adjacent": "Find indices i greater than zero where values[i] is smaller than values[i-1].",
        "palindrome": "Find left-half indices i whose value differs from the mirrored value values[-1-i].",
    }[mode]
    rules = [
        operation,
        "Return the number found."
        if result == "count"
        else "Return the found indices or values in ascending order.",
    ]
    return {
        "family": "pair_scan",
        "name": name,
        "question": _prompt(
            name,
            signature,
            rules,
            [f"{name}({x!r}) == {solve(x)!r}" for x in inputs[:2]],
        ),
        "response": "\n".join(lines),
        "tests": [f"assert {name}({x!r}) == {solve(x)!r}" for x in inputs],
        "graph": {
            "mode": mode,
            "parameter": parameter if mode == "difference" else None,
            "result": result,
        },
    }


def _rounded_affine(rng: random.Random, identity: int) -> dict[str, Any]:
    name = _function_name("rounded_affine", identity)
    scale = rng.choice((-5, -3, -2, -1, 1, 2, 3, 5))
    bias = rng.randint(-20, 20)
    divisor = rng.randint(2, 12)
    rounding = rng.choice(("floor", "ceil"))
    offset = rng.randint(-5, 5)
    absolute_input = bool(rng.getrandbits(1))

    def solve(n: int) -> int:
        value = abs(n) if absolute_input else n
        numerator = value * scale + bias
        quotient = (
            numerator // divisor if rounding == "floor" else -(-numerator // divisor)
        )
        return quotient + offset

    lines = [f"def {name}(n: int) -> int:"]
    lines.append("    value = abs(n)" if absolute_input else "    value = n")
    lines.append(f"    numerator = value * {scale} + ({bias})")
    lines.append(
        f"    quotient = numerator // {divisor}"
        if rounding == "floor"
        else f"    quotient = -(-numerator // {divisor})"
    )
    lines.append(f"    return quotient + ({offset})")
    inputs = [rng.randint(-200, 250) for _ in range(10)]
    signature = f"def {name}(n: int) -> int:"
    rules = [
        "Replace n by abs(n)." if absolute_input else "Use n with its original sign.",
        f"Compute numerator = n * {scale} + ({bias}) using that value.",
        f"Divide by {divisor} and round toward {'negative infinity' if rounding == 'floor' else 'positive infinity'}.",
        f"Add {offset} and return the integer result.",
    ]
    return {
        "family": "rounded_affine",
        "name": name,
        "question": _prompt(
            name,
            signature,
            rules,
            [f"{name}({x!r}) == {solve(x)!r}" for x in inputs[:2]],
        ),
        "response": "\n".join(lines),
        "tests": [f"assert {name}({x!r}) == {solve(x)!r}" for x in inputs],
        "graph": {
            "scale": scale,
            "bias": bias,
            "divisor": divisor,
            "rounding": rounding,
            "offset": offset,
            "absolute_input": absolute_input,
        },
    }


def _set_relation(rng: random.Random, identity: int) -> dict[str, Any]:
    name = _function_name("set_relation", identity)
    mode = rng.choice(("equal", "left_subset", "symmetric_difference"))
    ignore_case = bool(rng.getrandbits(1))
    ignore_spaces = bool(rng.getrandbits(1))

    def normalize(text: str) -> set[str]:
        if ignore_case:
            text = text.casefold()
        if ignore_spaces:
            text = "".join(text.split())
        return set(text)

    def solve(left: str, right: str) -> bool | int:
        left_set, right_set = normalize(left), normalize(right)
        if mode == "equal":
            return left_set == right_set
        if mode == "left_subset":
            return left_set <= right_set
        return len(left_set ^ right_set)

    annotation = "int" if mode == "symmetric_difference" else "bool"
    lines = [f"def {name}(left: str, right: str) -> {annotation}:"]
    if ignore_case:
        lines.append("    left, right = left.casefold(), right.casefold()")
    if ignore_spaces:
        lines.extend(
            ("    left = ''.join(left.split())", "    right = ''.join(right.split())")
        )
    lines.extend(
        (
            "    left_set, right_set = set(left), set(right)",
            {
                "equal": "    return left_set == right_set",
                "left_subset": "    return left_set <= right_set",
                "symmetric_difference": "    return len(left_set ^ right_set)",
            }[mode],
        )
    )
    alphabet = "aAbBcCxyzXYZ 012"
    inputs = [
        (
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 20))),
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 20))),
        )
        for _ in range(10)
    ]
    signature = f"def {name}(left: str, right: str) -> {annotation}:"
    rules = []
    if ignore_case:
        rules.append("Case-fold both strings.")
    if ignore_spaces:
        rules.append("Remove all whitespace from both strings.")
    rules.extend(
        (
            "Compare the sets of distinct remaining characters.",
            {
                "equal": "Return whether the two sets are equal.",
                "left_subset": "Return whether the left set is a subset of the right set.",
                "symmetric_difference": "Return the size of their symmetric difference.",
            }[mode],
        )
    )
    return {
        "family": "set_relation",
        "name": name,
        "question": _prompt(
            name,
            signature,
            rules,
            [f"{name}({a!r}, {b!r}) == {solve(a, b)!r}" for a, b in inputs[:2]],
        ),
        "response": "\n".join(lines),
        "tests": [f"assert {name}({a!r}, {b!r}) == {solve(a, b)!r}" for a, b in inputs],
        "graph": {
            "mode": mode,
            "ignore_case": ignore_case,
            "ignore_spaces": ignore_spaces,
        },
    }


GENERATORS: tuple[Callable[[random.Random, int], dict[str, Any]], ...] = (
    _index_rewrite,
    _frequency_filter,
    _nested_support,
    _sentence_scan,
    _pair_scan,
    _rounded_affine,
    _set_relation,
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
        raise FunctionGraphV2Error("generation shape differs")
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    seen: set[str] = set()
    local = 0
    while len(rows) < count:
        global_identity = shard_index + local * shard_count
        local += 1
        rng = random.Random(seed + global_identity * 1000003)
        task = GENERATORS[global_identity % len(GENERATORS)](rng, global_identity)
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
        verification_sha = hashlib.sha256(
            (task["response"] + "\n" + "\n".join(task["tests"])).encode()
        ).hexdigest()
        row = {
            **task,
            "schema": SCHEMA,
            "source": "shohin_failure_aligned_function_graph",
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
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "output": str(args.output.resolve()),
        "rows": len(rows),
        "seed": args.seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "ngram_width": args.ngram_width,
        "evaluation_sources": receipts,
        "counters": dict(sorted(counters.items())),
    }
    report["output_sha256"] = _atomic_lines(args.output, rows)
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
