"""Frozen-Shohin connected smoke for the multi-family raw compiler."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from source_deleted_multifamily_machine_board import build_frozen_board  # noqa: E402
from episode_functor_shohin_trunk import FrozenShohinTrunk  # noqa: E402
from multifamily_raw_machine_compiler import (  # noqa: E402
    SharedRawMachineCompiler,
    scan_query,
    scan_source,
)
from multifamily_shohin_feature_adapter import (  # noqa: E402
    PROTECTED_SHOHIN_SHA256,
    connected_feature_receipt,
    extract_query_unit_features,
    extract_source_unit_features,
)
from run_multifamily_raw_machine_smoke import (  # noqa: E402
    _collate,
    _evaluate,
    _role_loss,
)


def _combine_manifests(first: str, second: str) -> str:
    digest = sha256(b"MULTIFAMILY-COMBINED-FEATURE-MANIFEST-V1\0")
    digest.update(bytes.fromhex(first))
    digest.update(bytes.fromhex(second))
    return digest.hexdigest()


def _parse_blocks(value: str) -> tuple[int, ...]:
    try:
        blocks = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("block indices must be integers") from exc
    if not blocks or tuple(sorted(set(blocks))) != blocks:
        raise argparse.ArgumentTypeError(
            "block indices must be unique and increasing"
        )
    return blocks


def _extract(
    *,
    rows,
    trunk,
    tokenizer,
    device: torch.device,
    chunk_size: int,
):
    source, query, source_labels, query_labels = _collate(
        rows,
        device=device,
    )
    scanned_source = tuple(
        scan_source(row.candidate.source.encode("ascii")) for row in rows
    )
    scanned_query = tuple(
        scan_query(row.candidate.query.encode("ascii")) for row in rows
    )
    source_features, source_count, source_manifest = (
        extract_source_unit_features(
            trunk=trunk,
            tokenizer=tokenizer,
            scanned=scanned_source,
            batch=source,
            chunk_size=chunk_size,
        )
    )
    query_features, query_count, query_manifest = (
        extract_query_unit_features(
            trunk=trunk,
            tokenizer=tokenizer,
            scanned=scanned_query,
            batch=query,
            chunk_size=chunk_size,
        )
    )
    return {
        "query": query,
        "query_count": query_count,
        "query_features": query_features,
        "query_labels": query_labels,
        "query_manifest": query_manifest,
        "source": source,
        "source_count": source_count,
        "source_features": source_features,
        "source_labels": source_labels,
        "source_manifest": source_manifest,
    }


def run_connected_smoke(
    *,
    checkpoint: Path,
    tokenizer_path: Path,
    block_indices: tuple[int, ...],
    seed: int,
    steps: int,
    width: int,
    layers: int,
    learning_rate: float,
    feature_chunk_size: int,
    device: torch.device,
) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=PROTECTED_SHOHIN_SHA256,
        block_indices=block_indices,
        device=device,
    )
    checkpoint_receipt = trunk.parameter_receipt()
    board = build_frozen_board(
        seed=20260725,
        train_per_renderer=4,
        development_per_cell=2,
    )
    train_rows = [row for row in board if row.supervisor.split == "train"]
    development_rows = [
        row for row in board if row.supervisor.split == "development"
    ]
    train = _extract(
        rows=train_rows,
        trunk=trunk,
        tokenizer=tokenizer,
        device=device,
        chunk_size=feature_chunk_size,
    )
    development = _extract(
        rows=development_rows,
        trunk=trunk,
        tokenizer=tokenizer,
        device=device,
        chunk_size=feature_chunk_size,
    )
    feature_receipt = connected_feature_receipt(
        trunk=trunk,
        checkpoint_receipt=checkpoint_receipt,
        source_payload_count=(
            train["source_count"] + development["source_count"]
        ),
        query_payload_count=(
            train["query_count"] + development["query_count"]
        ),
        anonymous_source_manifest_sha256=_combine_manifests(
            train["source_manifest"],
            development["source_manifest"],
        ),
        anonymous_query_manifest_sha256=_combine_manifests(
            train["query_manifest"],
            development["query_manifest"],
        ),
    )
    model = SharedRawMachineCompiler(
        width=width,
        layers=layers,
        external_width=trunk.feature_width,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    losses: list[float] = []
    model.train()
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        source_output = model.compile_source(
            train["source"],
            external_unit_features=train["source_features"],
        )
        query_output = model.parse_query(
            train["query"],
            external_unit_features=train["query_features"],
        )
        loss = _role_loss(
            source_output.source_role_logits,
            query_output.query_role_logits,
            train["source_labels"],
            train["query_labels"],
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    report = {
        "block_indices": block_indices,
        "candidate_time_oracle_calls": 0,
        "candidate_time_search_calls": 0,
        "candidate_time_verifier_calls": 0,
        "development": _evaluate(
            model,
            development_rows,
            device=device,
            source_external=development["source_features"],
            query_external=development["query_features"],
        ),
        "device": str(device),
        "feature_receipt": asdict(feature_receipt),
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "parameter_receipt": asdict(model.parameter_receipt()),
        "seed": seed,
        "status": "frozen_shohin_connected_mechanics_smoke",
        "steps": steps,
        "train": _evaluate(
            model,
            train_rows,
            device=device,
            source_external=train["source_features"],
            query_external=train["query_features"],
        ),
    }
    del trunk
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "train/flagship_out/ckpt_0300000.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "artifacts/tokenizer/tokenizer.json",
    )
    parser.add_argument("--block-indices", type=_parse_blocks, default=(17, 25, 29))
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--feature-chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_connected_smoke(
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        block_indices=args.block_indices,
        seed=args.seed,
        steps=args.steps,
        width=args.width,
        layers=args.layers,
        learning_rate=args.learning_rate,
        feature_chunk_size=args.feature_chunk_size,
        device=torch.device(args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
