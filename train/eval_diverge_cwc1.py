#!/usr/bin/env python3
"""Evaluate DIVERGE-CWC1 whole-world commitment and matched controls."""

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

from diverge_cwc1_data import BOARD_ROWS, counterfactual_source, validate_record
from diverge_cwc1_runtime import (
    CWC1Config,
    CounterfactualWorldCommitter,
    module_state_sha256,
    tensorize_records,
)


SCHEMA = "shohin-diverge-cwc1-evaluation-v1"
TRAIN_SCHEMA = "shohin-diverge-cwc1-training-runtime-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_rows(path: Path, expected_sha256: str, split: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SystemExit("CWC1 board hash differs")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != BOARD_ROWS or any(row.get("split") != split for row in rows):
        raise SystemExit("CWC1 board geometry differs")
    for row in rows:
        validate_record(row)
    return rows


def _load_arm(
    root: Path,
    expected_arm: str,
    device: torch.device,
) -> tuple[CounterfactualWorldCommitter, dict[str, Any], str, str]:
    report_path = root / "report.json"
    checkpoint_path = root / "checkpoint_0001000.pt"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != TRAIN_SCHEMA or report.get("arm") != expected_arm:
        raise SystemExit(f"CWC1 training report differs for {expected_arm}")
    checkpoint_sha256 = sha256_path(checkpoint_path)
    if checkpoint_sha256 != report.get("checkpoint_sha256"):
        raise SystemExit(f"CWC1 checkpoint receipt differs for {expected_arm}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != TRAIN_SCHEMA
        or checkpoint.get("arm") != expected_arm
        or checkpoint.get("data_sha256") != report.get("data_sha256")
    ):
        raise SystemExit(f"CWC1 checkpoint metadata differs for {expected_arm}")
    config = CWC1Config(**checkpoint["config"])
    model = CounterfactualWorldCommitter(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if module_state_sha256(model) != checkpoint.get("model_state_sha256"):
        raise SystemExit(f"CWC1 model state differs for {expected_arm}")
    return model.eval(), report, checkpoint_sha256, sha256_path(report_path)


def _replace_symbols(text: str, mapping: Mapping[str, str]) -> str:
    alternatives = "|".join(
        re.escape(value) for value in sorted(mapping, key=len, reverse=True)
    )
    pattern = re.compile(r"(?<![a-z])(" + alternatives + r")(?![a-z])")
    return pattern.sub(lambda match: mapping[match.group(1)], text)


def _rename_word(identity: str, index: int) -> str:
    digest = hashlib.sha256(f"cwc1-rename:{identity}:{index}".encode("ascii")).digest()
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    return "".join(
        consonants[digest[2 * offset] % len(consonants)]
        + vowels[digest[2 * offset + 1] % len(vowels)]
        for offset in range(7)
    )


def _with_updated_source(row: Mapping[str, Any], source: str) -> dict[str, Any]:
    transformed = dict(row)
    transformed["source_text"] = source
    transformed["source_sha256"] = hashlib.sha256(source.encode("ascii")).hexdigest()
    transformed["counterfactual_sha256"] = hashlib.sha256(
        counterfactual_source(transformed).encode("ascii")
    ).hexdigest()
    return transformed


def _renamed(row: Mapping[str, Any]) -> dict[str, Any]:
    symbols = (*row["candidate_labels"], *row["aliases"], *row["registers"])
    mapping = {
        str(symbol): _rename_word(str(row["identity_sha256"]), index)
        for index, symbol in enumerate(symbols)
    }
    if len(set(mapping.values())) != len(mapping) or set(mapping) & set(mapping.values()):
        raise SystemExit("CWC1 rename map collides")
    transformed = dict(row)
    transformed["candidate_labels"] = [mapping[str(value)] for value in row["candidate_labels"]]
    transformed["aliases"] = [mapping[str(value)] for value in row["aliases"]]
    transformed["registers"] = [mapping[str(value)] for value in row["registers"]]
    return _with_updated_source(
        transformed,
        _replace_symbols(str(row["source_text"]), mapping),
    )


def _scrubbed(row: Mapping[str, Any]) -> dict[str, Any]:
    text = str(row["source_text"])
    left, right = (int(value) for value in row["directive_bounds"])
    return _with_updated_source(row, text[:left] + "#" * (right - left) + text[right:])


def _block_swapped(row: Mapping[str, Any]) -> dict[str, Any]:
    text = str(row["source_text"])
    bounds = [tuple(int(value) for value in item) for item in row["candidate_bounds"]]
    directive_bounds = tuple(int(value) for value in row["directive_bounds"])
    blocks = [text[left:right] for left, right in bounds]
    directive = text[directive_bounds[0] : directive_bounds[1]]
    if int(row["directive_position"]) == 0:
        source = f"{directive} {blocks[1]} {blocks[0]}"
        first_left = len(directive) + 1
        new_directive_bounds = (0, len(directive))
    else:
        source = f"{blocks[1]} {blocks[0]} {directive}"
        first_left = 0
        new_directive_bounds = (len(blocks[1]) + len(blocks[0]) + 2, len(source))
    new_bounds = []
    cursor = first_left
    for block in (blocks[1], blocks[0]):
        new_bounds.append([cursor, cursor + len(block)])
        cursor += len(block) + 1
    transformed = dict(row)
    transformed["directive_bounds"] = list(new_directive_bounds)
    transformed["candidate_bounds"] = new_bounds
    transformed["candidate_labels"] = list(reversed(row["candidate_labels"]))
    transformed["candidates"] = list(reversed(row["candidates"]))
    transformed["target_position"] = 1 - int(row["target_position"])
    return _with_updated_source(transformed, source)


@torch.no_grad()
def _predict(
    model: CounterfactualWorldCommitter,
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    normal_output = []
    counterfactual_output = []
    projection_residual = 0.0
    for start in range(0, len(rows), batch_size):
        subset = rows[start : start + batch_size]
        normal = tensorize_records(subset, device)
        partner = tensorize_records(subset, device, counterfactual=True)
        if arm == "involution":
            normal_scores = model.projected_scores(normal, partner)
            partner_scores = model.projected_scores(partner, normal)
            projection_residual = max(
                projection_residual,
                float((normal_scores - partner_scores.flip(dims=(-1,))).abs().max()),
            )
        else:
            normal_scores = model.raw_scores(normal)
            partner_scores = model.raw_scores(partner)
        normal_output.append(normal_scores.cpu())
        counterfactual_output.append(partner_scores.cpu())
    return (
        torch.cat(normal_output),
        torch.cat(counterfactual_output),
        projection_residual,
    )


def _score(
    rows: Sequence[Mapping[str, Any]],
    scores: torch.Tensor,
    *,
    flip_target: bool = False,
) -> dict[str, Any]:
    exact = 0
    margin_sum = 0.0
    renderer: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    predictions = scores.argmax(-1)
    for index, row in enumerate(rows):
        target = int(row["target_position"])
        if flip_target:
            target = 1 - target
        correct = int(predictions[index]) == target
        exact += correct
        margin_sum += float(scores[index, target] - scores[index, 1 - target])
        key = f"{row['renderer'][0]}:{row['renderer'][1]}"
        renderer[key][0] += correct
        renderer[key][1] += 1
    return {
        "rows": len(rows),
        "exact": exact,
        "exact_rate": exact / len(rows),
        "mean_signed_margin": margin_sum / len(rows),
        "renderer": {
            key: {"exact": value[0], "rows": value[1], "rate": value[0] / value[1]}
            for key, value in sorted(renderer.items())
        },
        "prediction_sha256": hashlib.sha256(predictions.numpy().tobytes()).hexdigest(),
    }


def evaluate_arm(
    model: CounterfactualWorldCommitter,
    rows: list[dict[str, Any]],
    *,
    arm: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    normal_scores, counterfactual_scores, residual = _predict(
        model, rows, arm=arm, device=device, batch_size=batch_size
    )
    renamed_rows = [_renamed(row) for row in rows]
    renamed_scores, _, _ = _predict(
        model, renamed_rows, arm=arm, device=device, batch_size=batch_size
    )
    swapped_rows = [_block_swapped(row) for row in rows]
    swapped_scores, _, _ = _predict(
        model, swapped_rows, arm=arm, device=device, batch_size=batch_size
    )
    scrubbed_rows = [_scrubbed(row) for row in rows]
    scrubbed_scores, _, _ = _predict(
        model, scrubbed_rows, arm=arm, device=device, batch_size=batch_size
    )
    return {
        "normal": _score(rows, normal_scores),
        "counterfactual": _score(rows, counterfactual_scores, flip_target=True),
        "entity_rename": _score(renamed_rows, renamed_scores),
        "block_swap": _score(swapped_rows, swapped_scores),
        "directive_scrub": _score(scrubbed_rows, scrubbed_scores),
        "projection_max_absolute_error": residual,
    }


def _minimum_renderer(result: Mapping[str, Any]) -> float:
    return min(value["rate"] for value in result["normal"]["renderer"].values())


def _matched(reports: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    fields = (
        "seed",
        "updates",
        "batch_size",
        "learning_rate",
        "data_sha256",
        "trainable_parameters",
        "initial_model_sha256",
        "source_bytes_seen",
        "forwards_per_update",
    )
    first = reports[0]
    return {
        field: all(report.get(field) == first.get(field) for report in reports[1:])
        for field in fields
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--board-sha256", required=True)
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--involution-root", type=Path, required=True)
    parser.add_argument("--duplicate-root", type=Path)
    parser.add_argument("--augmentation-root", type=Path)
    parser.add_argument("--development-result", type=Path)
    parser.add_argument("--development-result-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing CWC1 result: {args.output}")
    if args.split == "development":
        if args.duplicate_root is None or args.augmentation_root is None:
            raise SystemExit("CWC1 development requires both matched controls")
    else:
        if args.development_result is None or args.development_result_sha256 is None:
            raise SystemExit("CWC1 confirmation requires development authorization")
        if sha256_path(args.development_result) != args.development_result_sha256:
            raise SystemExit("CWC1 development authorization hash differs")
        authorization = json.loads(args.development_result.read_text(encoding="utf-8"))
        if authorization.get("all_pass") is not True:
            raise SystemExit("CWC1 development did not authorize confirmation")

    rows = _load_rows(args.board, args.board_sha256, args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = {"involution": args.involution_root}
    if args.split == "development":
        roots.update(
            {"duplicate": args.duplicate_root, "augmentation": args.augmentation_root}
        )
    arms = {}
    for name, root in roots.items():
        assert root is not None
        model, training, checkpoint_sha256, report_sha256 = _load_arm(root, name, device)
        arms[name] = {
            "root": str(root),
            "checkpoint_sha256": checkpoint_sha256,
            "training_report_sha256": report_sha256,
            "training": training,
            "evaluation": evaluate_arm(
                model, rows, arm=name, device=device, batch_size=args.batch_size
            ),
        }

    treatment = arms["involution"]
    treatment_eval = treatment["evaluation"]
    matched = _matched([arm["training"] for arm in arms.values()])
    gates = {
        "treatment_training_fit": treatment["training"]["training_evaluation"][
            "exact_rate"
        ]
        >= 0.99,
        "normal_exact": treatment_eval["normal"]["exact_rate"] >= 0.99,
        "counterfactual_exact": treatment_eval["counterfactual"]["exact_rate"] >= 0.99,
        "renderer_floor": _minimum_renderer(treatment_eval) >= 0.95,
        "entity_rename_exact": treatment_eval["entity_rename"]["exact_rate"] >= 0.99,
        "block_swap_exact": treatment_eval["block_swap"]["exact_rate"] >= 0.99,
        "directive_scrub_at_chance": 0.49
        <= treatment_eval["directive_scrub"]["exact_rate"]
        <= 0.51,
        "directive_scrub_zero_margin": abs(
            treatment_eval["directive_scrub"]["mean_signed_margin"]
        )
        <= 1e-6,
        "projection_identity_exact": treatment_eval["projection_max_absolute_error"] == 0.0,
        "matched_receipts": all(matched.values()),
    }
    if args.split == "development":
        duplicate = arms["duplicate"]
        augmentation = arms["augmentation"]
        strongest_control_normal = max(
            duplicate["evaluation"]["normal"]["exact_rate"],
            augmentation["evaluation"]["normal"]["exact_rate"],
        )
        gates.update(
            {
                "duplicate_training_fit": duplicate["training"]["training_evaluation"][
                    "exact_rate"
                ]
                >= 0.99,
                "augmentation_training_fit": augmentation["training"][
                    "training_evaluation"
                ]["exact_rate"]
                >= 0.99,
                "normal_within_one_point_of_strongest_control": treatment_eval["normal"][
                    "exact_rate"
                ]
                >= strongest_control_normal - 0.01,
            }
        )
    report = {
        "schema": SCHEMA,
        "split": args.split,
        "board": str(args.board),
        "board_sha256": args.board_sha256,
        "arms": arms,
        "matched_receipts": matched,
        "gates": gates,
        "all_pass": all(gates.values()),
        "confirmation_access_authorized": args.split == "development" and all(gates.values()),
        "claim_boundary": (
            "A pass qualifies a learned source-dependent whole-world selector on the "
            "frozen synthetic interface. It does not establish unrestricted language "
            "grounding or end-to-end reasoning."
        ),
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "all_pass": report["all_pass"],
                "gates": gates,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
