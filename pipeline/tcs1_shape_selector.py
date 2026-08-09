#!/usr/bin/env python3
"""Fit the leakage-resistant CPU shape control for TCS1."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from build_tcs1_candidate_sets import SCHEMA, sha256_file


MODEL_SCHEMA = "shohin-tcs1-shape-selector-v1"
REPORT_SCHEMA = "shohin-tcs1-shape-selection-report-v1"
FEATURE_NAMES = (
    "max_token_exhausted",
    "generated_token_fraction",
    "log_completion_chars",
    "log_completion_words",
    "log_question_chars",
    "completion_question_log_ratio",
    "unique_line_fraction",
    "unique_fourgram_fraction",
    "boxed_marker",
    "final_answer_marker",
    "therefore_marker",
    "self_correction_density",
    "hedge_density",
    "numeric_character_fraction",
    "equation_character_fraction",
    "question_numeric_fraction",
    "question_choice_marker",
    "question_science_marker",
)


class TCS1SelectorError(RuntimeError):
    """TCS1 selector data or output differs."""


def fraction(part: float, whole: float) -> float:
    return float(part) / max(float(whole), 1.0)


def marker_density(text: str, markers: tuple[str, ...]) -> float:
    lowered = text.casefold()
    return fraction(sum(lowered.count(marker) for marker in markers), len(text.split()))


def unique_fourgram_fraction(text: str) -> float:
    words = re.findall(r"\w+|[^\w\s]", text.casefold())
    grams = [tuple(words[index : index + 4]) for index in range(len(words) - 3)]
    return fraction(len(set(grams)), len(grams)) if grams else 1.0


def feature_vector(row: dict[str, Any]) -> list[float]:
    """Use text/shape only: no lineage, task label, gold, or peer outcome."""

    completion = str(row.get("completion") or "")
    question = str(row.get("question") or "")
    lines = [
        line.strip().casefold() for line in completion.splitlines() if line.strip()
    ]
    completion_log = math.log1p(len(completion))
    question_log = math.log1p(len(question))
    numeric_chars = sum(character.isdigit() for character in completion)
    equation_chars = sum(character in "=+-*/^<>" for character in completion)
    question_numeric = sum(character.isdigit() for character in question)
    return [
        float(bool(row.get("max_token_exhausted"))),
        fraction(float(row.get("generated_tokens") or 0), 1536),
        completion_log,
        math.log1p(len(completion.split())),
        question_log,
        completion_log - question_log,
        fraction(len(set(lines)), len(lines)) if lines else 1.0,
        unique_fourgram_fraction(completion),
        float(r"\boxed" in completion),
        float(bool(re.search(r"(?:final\s+)?answer\s*(?:is|:)", completion, re.I))),
        float("therefore" in completion.casefold()),
        marker_density(completion, ("wait", "mistake", "reconsider", "actually")),
        marker_density(completion, ("maybe", "perhaps", "i think", "likely")),
        fraction(numeric_chars, len(completion)),
        fraction(equation_chars, len(completion)),
        fraction(question_numeric, len(question)),
        float(bool(re.search(r"(?:^|\n)\s*(?:\(?[A-D]\)|[A-D][.:])", question))),
        float(
            bool(
                re.search(
                    r"\b(?:atom|molecule|cell|gene|protein|planet|force|energy|"
                    r"chemical|physical|biological|experiment|species|reaction)\b",
                    question,
                    re.I,
                )
            )
        ),
    ]


def load_groups(path: Path, split: str) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("schema") != SCHEMA or row.get("split") != split:
            raise TCS1SelectorError("candidate schema or split differs")
        identity = str(row.get("identity_sha256") or "")
        groups.setdefault(identity, []).append(row)
    if not groups:
        raise TCS1SelectorError("candidate set is empty")
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
        if [row["sample_index"] for row in rows] != [0, 1, 2]:
            raise TCS1SelectorError("candidate group geometry differs")
        if len({row["task"] for row in rows}) != 1:
            raise TCS1SelectorError("candidate task binding differs")
    return groups


class ShapeSelector(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), width),
            nn.GELU(),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def identity_validation(identity: str, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}\0{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 10 == 0


def materialize(
    groups: OrderedDict[str, list[dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[list[int]]]:
    features, labels, identities, indices = [], [], [], []
    for identity, rows in groups.items():
        local = []
        for row in rows:
            local.append(len(features))
            features.append(feature_vector(row))
            labels.append(float(row["correct"]))
            identities.append(identity)
        indices.append(local)
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        identities,
        indices,
    )


def selection_metrics(
    groups: OrderedDict[str, list[dict[str, Any]]], scores: torch.Tensor
) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = {"overall": Counter()}
    lineages: Counter[str] = Counter()
    offset = 0
    for rows in groups.values():
        local = scores[offset : offset + 3]
        winner = int(torch.argmax(local).item())
        offset += 3
        first, selected = rows[0], rows[winner]
        task = str(first["task"])
        counts.setdefault(task, Counter())
        for bucket in (counts["overall"], counts[task]):
            bucket["total"] += 1
            bucket["first_correct"] += int(first["correct"])
            bucket["selected_correct"] += int(selected["correct"])
            bucket["oracle_correct"] += int(any(row["correct"] for row in rows))
            bucket["repaired"] += int(not first["correct"] and selected["correct"])
            bucket["broken"] += int(first["correct"] and not selected["correct"])
        lineages[str(selected["lineage"])] += 1
    return {
        "buckets": {name: dict(value) for name, value in sorted(counts.items())},
        "selected_lineages": dict(sorted(lineages.items())),
    }


def train_one(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    identities: list[str],
    group_indices: list[list[int]],
    groups: OrderedDict[str, list[dict[str, Any]]],
    seed: int,
    width: int,
    epochs: int,
    patience_limit: int,
    shuffled: bool,
) -> tuple[ShapeSelector, torch.Tensor, torch.Tensor, dict[str, Any]]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 17)
    target = (
        labels[torch.randperm(len(labels), generator=generator)] if shuffled else labels
    )
    train_mask = torch.tensor(
        [not identity_validation(identity, seed) for identity in identities]
    )
    mean = features[train_mask].mean(dim=0)
    scale = features[train_mask].std(dim=0).clamp_min(1e-5)
    normalized = (features - mean) / scale
    model = ShapeSelector(width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    train_indices = torch.nonzero(train_mask, as_tuple=False).flatten()
    positive_weight = (1 - target[train_mask]).sum().clamp_min(1) / target[
        train_mask
    ].sum().clamp_min(1)
    pair_positive, pair_negative = [], []
    for local in group_indices:
        if not train_mask[local[0]]:
            continue
        positives = [index for index in local if target[index] == 1]
        negatives = [index for index in local if target[index] == 0]
        for positive in positives:
            for negative in negatives:
                pair_positive.append(positive)
                pair_negative.append(negative)
    if not pair_positive:
        raise TCS1SelectorError("training data has no mixed groups")
    positive_tensor = torch.tensor(pair_positive)
    negative_tensor = torch.tensor(pair_negative)
    best_state, best_score, patience = None, -1, 0
    validation_groups = OrderedDict(
        (identity, rows)
        for identity, rows in groups.items()
        if identity_validation(identity, seed)
    )
    validation_indices = [
        index
        for index, identity in enumerate(identities)
        if identity_validation(identity, seed)
    ]
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = model(normalized)
        bce = F.binary_cross_entropy_with_logits(
            scores[train_indices], target[train_indices], pos_weight=positive_weight
        )
        pairwise = F.softplus(
            -(scores[positive_tensor] - scores[negative_tensor])
        ).mean()
        (bce + pairwise).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_scores = model(normalized[validation_indices])
            metric = selection_metrics(validation_groups, validation_scores)["buckets"][
                "overall"
            ]["selected_correct"]
        if metric > best_score:
            best_score = metric
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            break
    if best_state is None:
        raise TCS1SelectorError("selector produced no state")
    model.load_state_dict(best_state)
    return (
        model,
        mean,
        scale,
        {"internal_validation_selected": best_score, "shuffled_labels": shuffled},
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise TCS1SelectorError("refusing existing selector output")
    train_groups = load_groups(args.train, "train")
    development_groups = load_groups(args.development, "development")
    if set(train_groups) & set(development_groups):
        raise TCS1SelectorError("train and development identities overlap")
    features, labels, identities, group_indices = materialize(train_groups)
    dev_features, _, _, _ = materialize(development_groups)
    model, mean, scale, fit_receipt = train_one(
        features=features,
        labels=labels,
        identities=identities,
        group_indices=group_indices,
        groups=train_groups,
        seed=args.seed,
        width=args.width,
        epochs=args.epochs,
        patience_limit=args.patience,
        shuffled=False,
    )
    shuffled_model, shuffled_mean, shuffled_scale, shuffled_receipt = train_one(
        features=features,
        labels=labels,
        identities=identities,
        group_indices=group_indices,
        groups=train_groups,
        seed=args.seed,
        width=args.width,
        epochs=args.epochs,
        patience_limit=args.patience,
        shuffled=True,
    )
    model.eval()
    shuffled_model.eval()
    with torch.no_grad():
        scores = model((dev_features - mean) / scale)
        shuffled_scores = shuffled_model(
            (dev_features - shuffled_mean) / shuffled_scale
        )
    args.output.mkdir(parents=True)
    checkpoint = args.output / "selector.pt"
    torch.save(
        {
            "schema": MODEL_SCHEMA,
            "feature_names": FEATURE_NAMES,
            "state_dict": model.state_dict(),
            "mean": mean,
            "scale": scale,
            "seed": args.seed,
            "train_sha256": sha256_file(args.train),
        },
        checkpoint,
    )
    metrics = selection_metrics(development_groups, scores)
    shuffled_metrics = selection_metrics(development_groups, shuffled_scores)
    overall = metrics["buckets"]["overall"]
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "feature_names": FEATURE_NAMES,
        "forbidden_features": ["lineage", "task", "gold", "correct", "peer_outcome"],
        "train": str(args.train.resolve()),
        "train_sha256": sha256_file(args.train),
        "development": str(args.development.resolve()),
        "development_sha256": sha256_file(args.development),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "fit": fit_receipt,
        "shuffled_fit": shuffled_receipt,
        "metrics": metrics,
        "shuffled_label_control": shuffled_metrics,
        "gates": {
            "oracle_ceiling_is_615": overall["oracle_correct"] == 615,
            "shape_reaches_603": overall["selected_correct"] >= 603,
            "shape_beats_depth_one_by_14": overall["selected_correct"] >= 603,
        },
    }
    temporary = args.output / "report.json.partial"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output / "report.json")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026080902)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=75)
    report = run(parser.parse_args())
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
