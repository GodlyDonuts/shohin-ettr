#!/usr/bin/env python3
"""Fit or apply a frozen-backbone candidate correctness head."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from product_candidate_reranker import CandidateReranker, FEATURE_NAMES, SCHEMA as SHAPE_SCHEMA
from select_product_self_consistency import choose_modal_candidate


FEATURE_SCHEMA = "shohin-product-candidate-features-v1"
MODEL_SCHEMA = "shohin-product-neural-candidate-reranker-v1"
REPORT_SCHEMA = "shohin-product-neural-candidate-selection-v1"


class NeuralRerankerError(RuntimeError):
    """Frozen candidate features or correctness-head artifacts differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_is_validation(identity: str, seed: int) -> bool:
    value = hashlib.sha256(f"{seed}\0{identity}".encode()).digest()
    return int.from_bytes(value[:8], "big") % 10 == 0


def load_feature_shards(
    paths: list[Path],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    hidden: list[torch.Tensor] = []
    shape: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    contract: dict[str, Any] | None = None
    candidate_sha256: str | None = None
    keys: set[tuple[str, int]] = set()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != FEATURE_SCHEMA:
            raise NeuralRerankerError("candidate feature schema differs")
        current = {
            "model_revision": payload["model_revision"],
            "adapter_sha256": payload["adapter_sha256"],
            "layer_offsets": tuple(payload["layer_offsets"]),
            "pooling": tuple(payload["pooling"]),
            "tail_tokens": int(payload["tail_tokens"]),
            "shape_feature_names": tuple(payload["shape_feature_names"]),
            "hidden_width": int(payload["hidden_features"].shape[1]),
        }
        if contract is None:
            contract = current
        elif current != contract:
            raise NeuralRerankerError("candidate feature shard contracts differ")
        if candidate_sha256 is None:
            candidate_sha256 = str(payload["candidate_sha256"])
        elif str(payload["candidate_sha256"]) != candidate_sha256:
            raise NeuralRerankerError("candidate feature sources differ within a set")
        if len(payload["metadata"]) != int(payload["hidden_features"].shape[0]):
            raise NeuralRerankerError("feature metadata length differs")
        for row in payload["metadata"]:
            key = (str(row["identity_sha256"]), int(row["sample_index"]))
            if key in keys:
                raise NeuralRerankerError("candidate feature rows overlap")
            keys.add(key)
            metadata.append(dict(row))
        hidden.append(payload["hidden_features"].to(torch.float16))
        shape.append(payload["shape_features"].to(torch.float32))
    if contract is None:
        raise NeuralRerankerError("no candidate feature shards were supplied")
    return torch.cat(hidden), torch.cat(shape), metadata, contract


def group_indices(metadata: list[dict[str, Any]]) -> OrderedDict[str, list[int]]:
    grouped: OrderedDict[str, list[int]] = OrderedDict()
    for index, row in enumerate(metadata):
        grouped.setdefault(str(row["identity_sha256"]), []).append(index)
    for indices in grouped.values():
        indices.sort(key=lambda index: int(metadata[index]["sample_index"]))
        if [int(metadata[index]["sample_index"]) for index in indices] != list(
            range(len(indices))
        ):
            raise NeuralRerankerError("candidate sample indices differ")
    return grouped


class FrozenFeatureCorrectnessHead(nn.Module):
    def __init__(self, hidden_width: int, shape_width: int, width: int) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_width)
        self.hidden_projection = nn.Linear(hidden_width, width)
        self.shape_projection = nn.Sequential(
            nn.LayerNorm(shape_width), nn.Linear(shape_width, max(width // 4, 16))
        )
        combined = width + max(width // 4, 16)
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Linear(combined, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, hidden: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        hidden_state = self.hidden_projection(self.hidden_norm(hidden))
        shape_state = self.shape_projection(shape)
        return self.output(torch.cat((hidden_state, shape_state), dim=-1)).squeeze(-1)


def score_all(
    model: FrozenFeatureCorrectnessHead,
    hidden: torch.Tensor,
    shape: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(hidden), batch_size):
            batch_hidden = hidden[offset : offset + batch_size].to(
                device=device, dtype=torch.float32
            )
            batch_shape = shape[offset : offset + batch_size].to(device=device)
            scores.append(model(batch_hidden, batch_shape).cpu())
    return torch.cat(scores)


def selection_metrics(
    grouped: OrderedDict[str, list[int]],
    metadata: list[dict[str, Any]],
    scores: torch.Tensor,
    *,
    validation: bool | None,
    seed: int,
) -> dict[str, float | int]:
    total = first = modal = oracle = selected = 0
    for identity, indices in grouped.items():
        is_validation = identity_is_validation(identity, seed)
        if validation is not None and validation != is_validation:
            continue
        rows = [metadata[index] for index in indices]
        row_scores = scores[indices].clone()
        for position, row in enumerate(rows):
            if bool(row.get("empty_completion")):
                row_scores[position] = -1e9
        winner = int(torch.argmax(row_scores).item())
        total += 1
        first += int(bool(rows[0]["correct"]))
        modal += int(bool(choose_modal_candidate(rows)["correct"]))
        oracle += int(any(bool(row["correct"]) for row in rows))
        selected += int(bool(rows[winner]["correct"]))
    return {
        "total": total,
        "first_correct": first,
        "first_accuracy": first / max(total, 1),
        "modal_correct": modal,
        "modal_accuracy": modal / max(total, 1),
        "oracle_correct": oracle,
        "oracle_accuracy": oracle / max(total, 1),
        "selected_correct": selected,
        "selected_accuracy": selected / max(total, 1),
    }


def shape_scores(shape: torch.Tensor, path: Path) -> tuple[torch.Tensor, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SHAPE_SCHEMA:
        raise NeuralRerankerError("shape reranker schema differs")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise NeuralRerankerError("shape reranker features differ")
    model = CandidateReranker(len(FEATURE_NAMES), int(payload["hidden_size"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    normalized = (shape - payload["feature_mean"]) / payload["feature_scale"]
    with torch.inference_mode():
        return model(normalized), sha256_file(path)


def build_pairs(
    grouped: OrderedDict[str, list[int]],
    metadata: list[dict[str, Any]],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positives: list[int] = []
    negatives: list[int] = []
    for identity, indices in grouped.items():
        if identity_is_validation(identity, seed):
            continue
        correct = [index for index in indices if bool(metadata[index]["correct"])]
        wrong = [index for index in indices if not bool(metadata[index]["correct"])]
        for positive in correct:
            for negative in wrong:
                positives.append(positive)
                negatives.append(negative)
    if not positives:
        raise NeuralRerankerError("training split has no mixed candidate groups")
    return torch.tensor(positives), torch.tensor(negatives)


def fit(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    hidden, shape, metadata, contract = load_feature_shards(args.features)
    grouped = group_indices(metadata)
    positive, negative = build_pairs(grouped, metadata, args.seed)
    device = torch.device(args.device)
    model = FrozenFeatureCorrectnessHead(
        hidden.shape[1], shape.shape[1], args.hidden_size
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = -1.0
    best_epoch = 0
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(positive), generator=generator)
        running_loss = 0.0
        for offset in range(0, len(order), args.batch_size):
            batch = order[offset : offset + args.batch_size]
            pos = positive[batch]
            neg = negative[batch]
            indices = torch.cat((pos, neg))
            batch_hidden = hidden[indices].to(device=device, dtype=torch.float32)
            batch_shape = shape[indices].to(device=device)
            scores = model(batch_hidden, batch_shape)
            pos_scores, neg_scores = scores.chunk(2)
            pairwise = F.softplus(-(pos_scores - neg_scores)).mean()
            bce = 0.5 * (
                F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
                + F.binary_cross_entropy_with_logits(
                    neg_scores, torch.zeros_like(neg_scores)
                )
            )
            loss = pairwise + args.bce_weight * bce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(batch)
        scores = score_all(model, hidden, shape, device, args.eval_batch_size)
        validation = selection_metrics(
            grouped, metadata, scores, validation=True, seed=args.seed
        )
        accuracy = float(validation["selected_accuracy"])
        if accuracy > best_validation:
            best_validation = accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        print(
            f"[neural-reranker] epoch={epoch} "
            f"loss={running_loss / len(order):.6f} "
            f"validation={accuracy:.6f} best={best_validation:.6f}",
            flush=True,
        )
        if patience >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    scores = score_all(model, hidden, shape, device, args.eval_batch_size)
    train_metrics = selection_metrics(
        grouped, metadata, scores, validation=False, seed=args.seed
    )
    validation_metrics = selection_metrics(
        grouped, metadata, scores, validation=True, seed=args.seed
    )
    baseline_scores, shape_sha = shape_scores(shape, args.shape_model)
    shape_validation = selection_metrics(
        grouped, metadata, baseline_scores, validation=True, seed=args.seed
    )
    payload = {
        "schema": MODEL_SCHEMA,
        "state_dict": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
        "hidden_width": int(hidden.shape[1]),
        "shape_width": int(shape.shape[1]),
        "head_width": args.hidden_size,
        "seed": args.seed,
        "feature_contract": contract,
        "feature_paths": [str(path.resolve()) for path in args.features],
        "feature_sha256": [sha256_file(path) for path in args.features],
        "shape_model": str(args.shape_model.resolve()),
        "shape_model_sha256": shape_sha,
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "shape_validation_metrics": shape_validation,
    }
    if args.output.exists():
        raise NeuralRerankerError(f"refusing existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    return {
        "schema": MODEL_SCHEMA,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "shape_validation_metrics": shape_validation,
    }


def select(args: argparse.Namespace) -> dict[str, Any]:
    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    if payload.get("schema") != MODEL_SCHEMA:
        raise NeuralRerankerError("neural reranker schema differs")
    hidden, shape, metadata, contract = load_feature_shards(args.features)
    if contract != payload["feature_contract"]:
        raise NeuralRerankerError("selection feature contract differs")
    model = FrozenFeatureCorrectnessHead(
        int(payload["hidden_width"]),
        int(payload["shape_width"]),
        int(payload["head_width"]),
    )
    model.load_state_dict(payload["state_dict"])
    device = torch.device(args.device)
    model.to(device)
    scores = score_all(model, hidden, shape, device, args.eval_batch_size)
    grouped = group_indices(metadata)
    metrics = selection_metrics(
        grouped, metadata, scores, validation=None, seed=int(payload["seed"])
    )
    results: list[dict[str, Any]] = []
    for identity, indices in grouped.items():
        row_scores = scores[indices].clone()
        for position, index in enumerate(indices):
            if bool(metadata[index].get("empty_completion")):
                row_scores[position] = -1e9
        winner = int(torch.argmax(row_scores).item())
        selected = metadata[indices[winner]]
        results.append(
            {
                "identity_sha256": identity,
                "task": selected["task"],
                "selected_sample_index": int(selected["sample_index"]),
                "selected_score": float(row_scores[winner]),
                "selected_prediction": selected.get("prediction"),
                "selected_correct": bool(selected["correct"]),
            }
        )
    report = {
        "schema": REPORT_SCHEMA,
        "selector_reads_gold": False,
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "feature_paths": [str(path.resolve()) for path in args.features],
        "feature_sha256": [sha256_file(path) for path in args.features],
        **metrics,
        "results": results,
    }
    if args.output.exists():
        raise NeuralRerankerError(f"refusing existing output: {args.output}")
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
    fit_parser.add_argument("--features", type=Path, action="append", required=True)
    fit_parser.add_argument("--shape-model", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--hidden-size", type=int, default=128)
    fit_parser.add_argument("--epochs", type=int, default=40)
    fit_parser.add_argument("--patience", type=int, default=8)
    fit_parser.add_argument("--batch-size", type=int, default=512)
    fit_parser.add_argument("--eval-batch-size", type=int, default=1024)
    fit_parser.add_argument("--learning-rate", type=float, default=3e-4)
    fit_parser.add_argument("--weight-decay", type=float, default=1e-3)
    fit_parser.add_argument("--bce-weight", type=float, default=0.25)
    fit_parser.add_argument("--seed", type=int, default=20260804)
    fit_parser.add_argument("--device", default="cuda:0")
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--features", type=Path, action="append", required=True)
    select_parser.add_argument("--model", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--eval-batch-size", type=int, default=1024)
    select_parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = fit(args) if args.command == "fit" else select(args)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
