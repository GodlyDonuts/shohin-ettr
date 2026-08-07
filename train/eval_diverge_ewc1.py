#!/usr/bin/env python3
"""Evaluate DIVERGE-EWC1 under frozen equivariance and source controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch

from diverge_ewc1_data import BOARD_ROWS, scan_integer_spans, validate_record
from diverge_ewc1_runtime import (
    EquivariantWorldCompiler,
    WorldCompilerConfig,
    hard_numeric_assignment,
    module_state_sha256,
    tensorize_worlds,
)


SCHEMA = "shohin-diverge-ewc1-evaluation-v1"
ALIAS_PERMUTATION = (3, 0, 6, 2, 7, 1, 5, 4)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_rows(path: Path, expected_sha256: str, split: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SystemExit("EWC1 board hash differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != BOARD_ROWS or any(row.get("split") != split for row in rows):
        raise SystemExit("EWC1 board geometry differs")
    for row in rows:
        validate_record(row)
    return rows


def _load_model(path: Path, expected_sha256: str, device: torch.device):
    if sha256_path(path) != expected_sha256:
        raise SystemExit("EWC1 checkpoint hash differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = WorldCompilerConfig(**checkpoint["config"])
    model = EquivariantWorldCompiler(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if module_state_sha256(model) != checkpoint["model_state_sha256"]:
        raise SystemExit("EWC1 model state hash differs")
    return model.eval(), checkpoint


def _replace_symbols(text: str, mapping: Mapping[str, str]) -> str:
    pattern = re.compile(
        r"(?<![a-z])(" + "|".join(re.escape(value) for value in sorted(mapping, key=len, reverse=True)) + r")(?![a-z])"
    )
    return pattern.sub(lambda match: mapping[match.group(1)], text)


def _renamed(row: Mapping[str, Any]) -> dict[str, Any]:
    symbols = (*row["aliases"], *row["registers"])
    mapping = {}
    for index, symbol in enumerate(symbols):
        digest = hashlib.sha256(f"ewc1-rename:{row['identity_sha256']}:{index}".encode("ascii")).digest()
        consonants = "bcdfghjklmnpqrstvwxyz"
        vowels = "aeiou"
        mapping[str(symbol)] = "".join(
            consonants[digest[2 * offset] % len(consonants)]
            + vowels[digest[2 * offset + 1] % len(vowels)]
            for offset in range(6)
        )
    transformed = dict(row)
    transformed["source_text"] = _replace_symbols(str(row["source_text"]), mapping)
    transformed["source_sha256"] = hashlib.sha256(transformed["source_text"].encode("ascii")).hexdigest()
    transformed["aliases"] = [mapping[str(value)] for value in row["aliases"]]
    transformed["registers"] = [mapping[str(value)] for value in row["registers"]]
    return transformed


def _scrubbed(row: Mapping[str, Any]) -> dict[str, Any]:
    text = str(row["source_text"])
    keep = [False] * len(text)
    for left, right in scan_integer_spans(text):
        keep[left:right] = [True] * (right - left)
    symbols = (*row["aliases"], *row["registers"])
    for symbol in symbols:
        for match in re.finditer(rf"(?<![a-z]){re.escape(str(symbol))}(?![a-z])", text):
            keep[match.start() : match.end()] = [True] * (match.end() - match.start())
    scrubbed = "".join(character if keep[index] else "#" for index, character in enumerate(text))
    transformed = dict(row)
    transformed["source_text"] = scrubbed
    transformed["source_sha256"] = hashlib.sha256(scrubbed.encode("ascii")).hexdigest()
    return transformed


def _register_swapped(row: Mapping[str, Any]) -> dict[str, Any]:
    transformed = dict(row)
    transformed["registers"] = list(reversed(row["registers"]))
    return transformed


def _alias_permuted(row: Mapping[str, Any]) -> dict[str, Any]:
    transformed = dict(row)
    transformed["aliases"] = [row["aliases"][index] for index in ALIAS_PERMUTATION]
    return transformed


@torch.no_grad()
def _predict(
    model: EquivariantWorldCompiler,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> list[tuple[tuple[int, int], tuple[int, ...]]]:
    outputs = []
    for start in range(0, len(rows), batch_size):
        subset = rows[start : start + batch_size]
        batch = tensorize_worlds(subset, device)
        numeric_logits, operation_logits = model(batch)
        for index, row in enumerate(subset):
            text = str(row["source_text"])
            numeric_spans = scan_integer_spans(text)
            assignment = hard_numeric_assignment(
                numeric_logits[index], int(batch.numeric_mask[index].sum())
            )
            initial = tuple(
                int(text[numeric_spans[value][0] : numeric_spans[value][1]])
                for value in assignment
            )
            alias_count = int(batch.alias_mask[index].sum())
            symbols = tuple(
                int(batch.alias_group_ids[index, position])
                for position, logit in enumerate(operation_logits[index, :alias_count])
                if float(logit) >= 0.0
            )
            outputs.append(((initial[0], initial[1]), symbols))
    return outputs


def _score(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[tuple[tuple[int, int], tuple[int, ...]]],
) -> dict[str, Any]:
    initial_exact = 0
    operation_exact = 0
    joint_exact = 0
    renderers: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row, prediction in zip(rows, predictions, strict=True):
        initial_ok = prediction[0] == tuple(int(value) for value in row["initial_state"])
        operation_ok = prediction[1] == tuple(int(value) for value in row["symbols"])
        exact = initial_ok and operation_ok
        initial_exact += initial_ok
        operation_exact += operation_ok
        joint_exact += exact
        renderer = f"{row['renderer'][0]}:{row['renderer'][1]}"
        renderers[renderer][0] += exact
        renderers[renderer][1] += 1
    return {
        "rows": len(rows),
        "initial_exact": initial_exact,
        "initial_exact_rate": initial_exact / len(rows),
        "operation_exact": operation_exact,
        "operation_exact_rate": operation_exact / len(rows),
        "joint_exact": joint_exact,
        "joint_exact_rate": joint_exact / len(rows),
        "renderer": {
            key: {"exact": value[0], "rows": value[1], "rate": value[0] / value[1]}
            for key, value in sorted(renderers.items())
        },
    }


def evaluate_model(
    model: EquivariantWorldCompiler,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    normal = _predict(model, rows, device=device, batch_size=batch_size)
    register_rows = [_register_swapped(row) for row in rows]
    register_raw = _predict(model, register_rows, device=device, batch_size=batch_size)
    register_mapped = [((value[0][1], value[0][0]), value[1]) for value in register_raw]
    alias_rows = [_alias_permuted(row) for row in rows]
    alias_raw = _predict(model, alias_rows, device=device, batch_size=batch_size)
    alias_mapped = []
    for row, transformed, prediction in zip(rows, alias_rows, alias_raw, strict=True):
        original_index = {value: index for index, value in enumerate(row["aliases"])}
        alias_mapped.append(
            (
                prediction[0],
                tuple(original_index[transformed["aliases"][index]] for index in prediction[1]),
            )
        )
    renamed_rows = [_renamed(row) for row in rows]
    renamed = _predict(model, renamed_rows, device=device, batch_size=batch_size)
    scrubbed_rows = [_scrubbed(row) for row in rows]
    scrubbed = _predict(model, scrubbed_rows, device=device, batch_size=batch_size)
    return {
        "normal": _score(rows, normal),
        "register_order_mapped": _score(rows, register_mapped),
        "alias_order_mapped": _score(rows, alias_mapped),
        "entity_rename": _score(rows, renamed),
        "source_scrub": _score(rows, scrubbed),
        "normal_prediction_sha256": hashlib.sha256(json.dumps(normal, sort_keys=True).encode()).hexdigest(),
    }


def _minimum_renderer(result: Mapping[str, Any]) -> float:
    return min(value["rate"] for value in result["normal"]["renderer"].values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--board-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--control-checkpoint", type=Path)
    parser.add_argument("--control-checkpoint-sha256")
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--training-report-sha256", required=True)
    parser.add_argument("--control-training-report", type=Path)
    parser.add_argument("--control-training-report-sha256")
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--development-result", type=Path)
    parser.add_argument("--development-result-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing EWC1 result: {args.output}")
    if args.split == "development":
        if not args.control_checkpoint or not args.control_checkpoint_sha256:
            raise SystemExit("EWC1 development requires the matched control")
    else:
        if not args.development_result or not args.development_result_sha256:
            raise SystemExit("EWC1 confirmation requires development authorization")
        if sha256_path(args.development_result) != args.development_result_sha256:
            raise SystemExit("EWC1 development authorization hash differs")
        authorization = json.loads(args.development_result.read_text())
        if not authorization.get("all_pass"):
            raise SystemExit("EWC1 development did not authorize confirmation")

    if sha256_path(args.training_report) != args.training_report_sha256:
        raise SystemExit("EWC1 training report hash differs")
    training_report = json.loads(args.training_report.read_text())
    rows = _load_rows(args.board, args.board_sha256, args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_model(args.checkpoint, args.checkpoint_sha256, device)
    treatment = evaluate_model(model, rows, device=device, batch_size=args.batch_size)
    control = None
    control_report = None
    if args.split == "development":
        if not args.control_training_report or not args.control_training_report_sha256:
            raise SystemExit("EWC1 development control report is absent")
        if sha256_path(args.control_training_report) != args.control_training_report_sha256:
            raise SystemExit("EWC1 control training report hash differs")
        control_report = json.loads(args.control_training_report.read_text())
        control_model, control_checkpoint = _load_model(
            args.control_checkpoint, args.control_checkpoint_sha256, device
        )
        control = evaluate_model(control_model, rows, device=device, batch_size=args.batch_size)
        parameter_match = training_report["trainable_parameters"] == control_report["trainable_parameters"]
        data_match = training_report["data_sha256"] == control_report["data_sha256"]
        schedule_match = all(
            training_report[key] == control_report[key]
            for key in ("updates", "batch_size", "learning_rate", "seed")
        )
    else:
        control_checkpoint = None
        parameter_match = True
        data_match = True
        schedule_match = True

    gates = {
        "treatment_training_fit": training_report["training_evaluation"]["joint_exact_rate"] >= 0.99,
        "normal_joint": treatment["normal"]["joint_exact_rate"] >= 0.99,
        "normal_initial": treatment["normal"]["initial_exact_rate"] >= 0.995,
        "normal_operations": treatment["normal"]["operation_exact_rate"] >= 0.99,
        "renderer_floor": _minimum_renderer(treatment) >= 0.95,
        "register_order_mapped": treatment["register_order_mapped"]["joint_exact_rate"] >= 0.99,
        "alias_order_mapped": treatment["alias_order_mapped"]["joint_exact_rate"] >= 0.99,
        "entity_rename": treatment["entity_rename"]["joint_exact_rate"] >= 0.99,
        "source_scrub_collapse": treatment["source_scrub"]["joint_exact_rate"] <= 0.20,
        "parameter_match": parameter_match,
        "data_match": data_match,
        "schedule_match": schedule_match,
    }
    if control is not None:
        gates.update(
            {
                "control_training_fit": control_report["training_evaluation"]["joint_exact_rate"] >= 0.99,
                "control_normal": control["normal"]["joint_exact_rate"] >= 0.95,
                "equivariance_advantage": (
                    treatment["register_order_mapped"]["joint_exact_rate"]
                    - control["register_order_mapped"]["joint_exact_rate"]
                    >= 0.25
                ),
            }
        )
    report = {
        "schema": SCHEMA,
        "split": args.split,
        "board": str(args.board),
        "board_sha256": args.board_sha256,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "model_state_sha256": checkpoint["model_state_sha256"],
        "control_checkpoint": None if args.control_checkpoint is None else str(args.control_checkpoint),
        "treatment": treatment,
        "control": control,
        "gates": gates,
        "all_pass": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "output_sha256": sha256_path(args.output), "all_pass": report["all_pass"], "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
