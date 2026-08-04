#!/usr/bin/env python3
"""Train or apply a label-blind-at-inference candidate reranker.

The model is intentionally small and operates only on candidate agreement and
completion-shape metadata.  It is a fast control for deciding whether the
measured pass@K gap requires semantic verification.  Correctness labels are
used only by ``fit`` and by post-selection reporting; feature extraction and
candidate selection never inspect gold or correctness.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from select_product_self_consistency import canonical_prediction, choose_modal_candidate


SCHEMA = "shohin-product-candidate-reranker-v1"
REPORT_SCHEMA = "shohin-product-candidate-reranker-selection-v1"


class CandidateRerankerError(RuntimeError):
    """Candidate rows or a reranker artifact violate the fixed contract."""


FEATURE_NAMES = (
    "sample_position",
    "answer_vote_fraction",
    "answer_is_mode",
    "unique_answer_fraction",
    "prediction_present",
    "explicit_final_answer",
    "max_token_exhausted",
    "draft_max_token_exhausted",
    "used_finalization",
    "generated_token_fraction",
    "finalization_token_fraction",
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
    "prediction_near_tail",
    "question_numeric_fraction",
    "question_choice_marker",
    "question_science_marker",
    "task_gsm8k",
    "task_math500",
    "task_aime",
    "task_short_answer",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise CandidateRerankerError(f"missing candidate source: {path}")
        try:
            rows.extend(
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            )
        except json.JSONDecodeError as exc:
            raise CandidateRerankerError(f"malformed candidate source: {path}") from exc
    return rows


def group_candidates(
    candidates: list[dict[str, Any]],
) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in candidates:
        identity = str(row.get("identity_sha256") or "")
        if not identity:
            raise CandidateRerankerError("candidate identity is missing")
        grouped.setdefault(identity, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
        indices = [int(row["sample_index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise CandidateRerankerError("candidate sample indices differ")
        task = str(rows[0]["task"])
        if any(str(row["task"]) != task for row in rows):
            raise CandidateRerankerError("candidate tasks differ within a group")
    return grouped


def _fraction(part: float, whole: float) -> float:
    return float(part) / max(float(whole), 1.0)


def _unique_fourgram_fraction(text: str) -> float:
    words = re.findall(r"\w+|[^\w\s]", text.casefold())
    grams = [tuple(words[index : index + 4]) for index in range(len(words) - 3)]
    return _fraction(len(set(grams)), len(grams)) if grams else 1.0


def _marker_density(text: str, markers: tuple[str, ...]) -> float:
    lowered = text.casefold()
    return _fraction(sum(lowered.count(marker) for marker in markers), len(text.split()))


def feature_vector(
    row: dict[str, Any], group: list[dict[str, Any]]
) -> list[float]:
    """Return label-free candidate features; never inspect ``gold`` or ``correct``."""

    task = str(row["task"])
    canonical = [canonical_prediction(task, item.get("prediction")) for item in group]
    counts = Counter(value for value in canonical if value is not None)
    candidate_answer = canonical[int(row["sample_index"])]
    vote_count = counts.get(candidate_answer, 0) if candidate_answer is not None else 0
    mode_count = max(counts.values(), default=0)
    completion = str(row.get("completion") or "")
    question = str(row.get("question") or "")
    lines = [line.strip().casefold() for line in completion.splitlines() if line.strip()]
    words = completion.split()
    prediction = str(row.get("prediction") or "").strip()
    tail = completion[-max(256, len(prediction) * 4) :].casefold()
    numeric_chars = sum(char.isdigit() for char in completion)
    equation_chars = sum(char in "=+-*/^<>" for char in completion)
    question_numeric = sum(char.isdigit() for char in question)
    choice_marker = bool(re.search(r"(?:^|\n)\s*(?:\(?[A-D]\)|[A-D][.:])", question))
    science_marker = bool(
        re.search(
            r"\b(?:atom|molecule|cell|gene|protein|planet|force|energy|"
            r"chemical|physical|biological|experiment|species|reaction)\b",
            question,
            flags=re.IGNORECASE,
        )
    )
    completion_log = math.log1p(len(completion))
    question_log = math.log1p(len(question))
    return [
        _fraction(int(row["sample_index"]), len(group) - 1),
        _fraction(vote_count, len(group)),
        float(candidate_answer is not None and vote_count == mode_count),
        _fraction(len(counts), len(group)),
        float(candidate_answer is not None),
        float(bool(row.get("explicit_final_answer"))),
        float(bool(row.get("max_token_exhausted"))),
        float(bool(row.get("draft_max_token_exhausted"))),
        float(row.get("finalization") is not None),
        _fraction(float(row.get("generated_tokens") or 0), 1536),
        _fraction(float(row.get("finalization_generated_tokens") or 0), 64),
        completion_log,
        math.log1p(len(words)),
        question_log,
        completion_log - question_log,
        _fraction(len(set(lines)), len(lines)) if lines else 1.0,
        _unique_fourgram_fraction(completion),
        float(r"\boxed" in completion),
        float(bool(re.search(r"(?:final\s+)?answer\s*(?:is|:)", completion, re.I))),
        float("therefore" in completion.casefold()),
        _marker_density(completion, ("wait", "mistake", "reconsider", "actually")),
        _marker_density(completion, ("maybe", "perhaps", "i think", "likely")),
        _fraction(numeric_chars, len(completion)),
        _fraction(equation_chars, len(completion)),
        float(bool(prediction) and prediction.casefold() in tail),
        _fraction(question_numeric, len(question)),
        float(choice_marker),
        float(science_marker),
        float(task == "gsm8k"),
        float(task == "math500"),
        float(task == "aime"),
        float(task in {"bbh_logic", "gpqa"}),
    ]


class CandidateReranker(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_count, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


def _materialize(
    grouped: OrderedDict[str, list[dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[list[int]]]:
    features: list[list[float]] = []
    labels: list[float] = []
    identities: list[str] = []
    group_indices: list[list[int]] = []
    for identity, rows in grouped.items():
        indices: list[int] = []
        for row in rows:
            indices.append(len(features))
            features.append(feature_vector(row, rows))
            labels.append(float(bool(row["correct"])))
            identities.append(identity)
        group_indices.append(indices)
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        identities,
        group_indices,
    )


def _identity_is_validation(identity: str, seed: int) -> bool:
    value = hashlib.sha256(f"{seed}\0{identity}".encode()).digest()
    return int.from_bytes(value[:8], "big") % 10 == 0


def _selection_metrics(
    grouped: OrderedDict[str, list[dict[str, Any]]],
    scores: torch.Tensor,
    identities: list[str],
    validation: bool | None,
    seed: int,
) -> dict[str, float | int]:
    offset = 0
    total = first = modal = oracle = selected = 0
    for identity, rows in grouped.items():
        row_scores = scores[offset : offset + len(rows)]
        offset += len(rows)
        is_validation = _identity_is_validation(identity, seed)
        if validation is not None and validation != is_validation:
            continue
        winner = int(torch.argmax(row_scores).item())
        total += 1
        first += int(bool(rows[0]["correct"]))
        modal += int(bool(choose_modal_candidate(rows)["correct"]))
        oracle += int(any(bool(row["correct"]) for row in rows))
        selected += int(bool(rows[winner]["correct"]))
    return {
        "total": total,
        "first_correct": first,
        "first_accuracy": _fraction(first, total),
        "modal_correct": modal,
        "modal_accuracy": _fraction(modal, total),
        "oracle_correct": oracle,
        "oracle_accuracy": _fraction(oracle, total),
        "selected_correct": selected,
        "selected_accuracy": _fraction(selected, total),
    }


def fit(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    candidates = load_candidates(args.candidates)
    grouped = group_candidates(candidates)
    features, labels, identities, group_indices = _materialize(grouped)
    train_mask = torch.tensor(
        [not _identity_is_validation(identity, args.seed) for identity in identities]
    )
    if not train_mask.any() or train_mask.all():
        raise CandidateRerankerError("identity split does not contain train and validation")
    mean = features[train_mask].mean(dim=0)
    scale = features[train_mask].std(dim=0).clamp_min(1e-5)
    normalized = (features - mean) / scale
    model = CandidateReranker(len(FEATURE_NAMES), args.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    train_indices = torch.nonzero(train_mask, as_tuple=False).flatten()
    positives = labels[train_mask].sum().clamp_min(1.0)
    negatives = (1.0 - labels[train_mask]).sum().clamp_min(1.0)
    positive_weight = negatives / positives
    pair_positive: list[int] = []
    pair_negative: list[int] = []
    for indices in group_indices:
        if not train_mask[indices[0]]:
            continue
        positive_rows = [index for index in indices if labels[index] == 1]
        negative_rows = [index for index in indices if labels[index] == 0]
        for positive in positive_rows:
            for negative in negative_rows:
                pair_positive.append(positive)
                pair_negative.append(negative)
    if not pair_positive:
        raise CandidateRerankerError("training split has no mixed candidate groups")
    pair_positive_tensor = torch.tensor(pair_positive)
    pair_negative_tensor = torch.tensor(pair_negative)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = -1.0
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = model(normalized)
        bce = F.binary_cross_entropy_with_logits(
            scores[train_indices],
            labels[train_indices],
            pos_weight=positive_weight,
        )
        pairwise = F.softplus(
            -(scores[pair_positive_tensor] - scores[pair_negative_tensor])
        ).mean()
        loss = bce + args.pairwise_weight * pairwise
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            all_scores = model(normalized)
            validation = _selection_metrics(grouped, all_scores, identities, True, args.seed)
        accuracy = float(validation["selected_accuracy"])
        if accuracy > best_validation:
            best_validation = accuracy
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if epoch % 25 == 0 or epoch == 1:
            print(
                f"[candidate-reranker] epoch={epoch} loss={float(loss):.5f} "
                f"validation={accuracy:.5f} best={best_validation:.5f}",
                flush=True,
            )
        if patience >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    with torch.no_grad():
        final_scores = model(normalized)
    train_metrics = _selection_metrics(grouped, final_scores, identities, False, args.seed)
    validation_metrics = _selection_metrics(grouped, final_scores, identities, True, args.seed)
    payload = {
        "schema": SCHEMA,
        "feature_names": FEATURE_NAMES,
        "hidden_size": args.hidden_size,
        "state_dict": model.state_dict(),
        "feature_mean": mean,
        "feature_scale": scale,
        "seed": args.seed,
        "source_paths": [str(path.resolve()) for path in args.candidates],
        "source_sha256": [_sha256(path) for path in args.candidates],
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }
    if args.output.exists():
        raise CandidateRerankerError(f"refusing existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    return {
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }


def select(args: argparse.Namespace) -> dict[str, Any]:
    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA or tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise CandidateRerankerError("reranker schema or features differ")
    model = CandidateReranker(len(FEATURE_NAMES), int(payload["hidden_size"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    candidates = load_candidates(args.candidates)
    grouped = group_candidates(candidates)
    features, _, identities, _ = _materialize(grouped)
    normalized = (features - payload["feature_mean"]) / payload["feature_scale"]
    with torch.no_grad():
        scores = model(normalized)
    metrics = _selection_metrics(grouped, scores, identities, None, int(payload["seed"]))
    results: list[dict[str, Any]] = []
    offset = 0
    for identity, rows in grouped.items():
        row_scores = scores[offset : offset + len(rows)]
        winner = int(torch.argmax(row_scores).item())
        offset += len(rows)
        selected = rows[winner]
        results.append(
            {
                "identity_sha256": identity,
                "task": selected["task"],
                "selected_sample_index": int(selected["sample_index"]),
                "selected_score": float(row_scores[winner]),
                "selected_prediction": selected.get("prediction"),
                "selected_completion": selected.get("completion"),
                "selected_correct": bool(selected["correct"]),
            }
        )
    report = {
        "schema": REPORT_SCHEMA,
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "candidate_paths": [str(path.resolve()) for path in args.candidates],
        "candidate_sha256": [_sha256(path) for path in args.candidates],
        "selector_reads_gold": False,
        **metrics,
        "results": results,
    }
    if args.output.exists():
        raise CandidateRerankerError(f"refusing existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    return {key: value for key, value in report.items() if key != "results"}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--candidates", type=Path, action="append", required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--hidden-size", type=int, default=64)
    fit_parser.add_argument("--epochs", type=int, default=500)
    fit_parser.add_argument("--patience", type=int, default=75)
    fit_parser.add_argument("--learning-rate", type=float, default=3e-3)
    fit_parser.add_argument("--pairwise-weight", type=float, default=1.0)
    fit_parser.add_argument("--seed", type=int, default=20260804)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--model", type=Path, required=True)
    select_parser.add_argument("--candidates", type=Path, action="append", required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        report = fit(args)
    else:
        report = select(args)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
