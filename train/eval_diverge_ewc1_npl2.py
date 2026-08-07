#!/usr/bin/env python3
"""Replace only NPL2's structural WORLD parser with confirmed EWC1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

import eval_diverge_npl2_development as npl2
from diverge_npl2_runtime import TypedEpisode, TypedProgram
from diverge_ewc1_runtime import (
    EquivariantWorldCompiler,
    WorldCompilerConfig,
    module_state_sha256,
)
from eval_diverge_ewc1 import _predict, sha256_path


DEVELOPMENT_SCHEMA = "shohin-diverge-ewc1-npl2-development-v1"
CONFIRMATION_SCHEMA = "shohin-diverge-ewc1-npl2-confirmation-seed-v1"


def build_typed_cache(
    public: Sequence[Mapping[str, Any]],
    predictions: Sequence[tuple[tuple[int, int], tuple[int, ...]]],
) -> dict[str, TypedEpisode]:
    expected = sum(
        len(record["acquisition"]) + len(record["transfer"]) for record in public
    )
    if len(predictions) != expected:
        raise RuntimeError("EWC1/NPL2 prediction geometry differs")
    cursor = 0
    cache = {}
    for record in public:
        acquisition = []
        transfer = []
        for source, destination in (
            (record["acquisition"], acquisition),
            (record["transfer"], transfer),
        ):
            for _ in source:
                initial, symbols = predictions[cursor]
                cursor += 1
                if not symbols:
                    raise RuntimeError("EWC1/NPL2 compiled an empty program")
                destination.append(
                    TypedProgram(initial_state=initial, symbols=symbols)
                )
        episode_id = str(record["episode_id"])
        if episode_id in cache:
            raise RuntimeError("EWC1/NPL2 episode identity repeats")
        cache[episode_id] = TypedEpisode(
            episode_id=episode_id,
            branch_names=tuple(str(value) for value in record["branch_names"]),
            acquisition=tuple(acquisition),
            transfer=tuple(transfer),
        )
    if cursor != len(predictions):
        raise RuntimeError("EWC1/NPL2 prediction cursor differs")
    return cache


@torch.no_grad()
def compile_typed_cache(
    model: EquivariantWorldCompiler,
    public: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, TypedEpisode]:
    work = []
    for record in public:
        aliases = [str(value) for value in record["aliases"]]
        registers = [str(value) for value in record["register_names"]]
        for program in (*record["acquisition"], *record["transfer"]):
            work.append(
                {
                    "source_text": str(program["source_text"]),
                    "source_sha256": str(program["source_sha256"]),
                    "aliases": aliases,
                    "registers": registers,
                }
            )
    predictions = _predict(model, work, device=device, batch_size=batch_size)
    return build_typed_cache(public, predictions)


def _option(arguments: Sequence[str], name: str) -> str:
    try:
        index = arguments.index(name)
        return arguments[index + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit(f"EWC1/NPL2 missing base option {name}") from error


def _load_world_model(
    checkpoint_path: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[EquivariantWorldCompiler, dict[str, Any]]:
    if sha256_path(checkpoint_path) != expected_sha256:
        raise SystemExit("EWC1/NPL2 WORLD checkpoint hash differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = WorldCompilerConfig(**checkpoint["config"])
    if config.mode != "equivariant":
        raise SystemExit("EWC1/NPL2 requires the equivariant compiler")
    model = EquivariantWorldCompiler(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if module_state_sha256(model) != checkpoint["model_state_sha256"]:
        raise SystemExit("EWC1/NPL2 WORLD model state hash differs")
    return model.eval(), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ewc-checkpoint", type=Path, required=True)
    parser.add_argument("--ewc-checkpoint-sha256", required=True)
    parser.add_argument("--ewc-result", type=Path, required=True)
    parser.add_argument("--ewc-result-sha256", required=True)
    parser.add_argument("--ewc-batch-size", type=int, default=512)
    args, remaining = parser.parse_known_args()
    if sha256_path(args.ewc_result) != args.ewc_result_sha256:
        raise SystemExit("EWC1/NPL2 qualification result hash differs")
    result = json.loads(args.ewc_result.read_text())
    if not result.get("all_pass") or result.get("split") not in (
        "development",
        "confirmation",
    ):
        raise SystemExit("EWC1/NPL2 requires a passing EWC1 result")
    if not torch.cuda.is_available():
        raise SystemExit("EWC1/NPL2 integration requires CUDA")

    public_path = Path(_option(remaining, "--public-data"))
    public_sha256 = _option(remaining, "--public-data-sha256")
    public = npl2._load_jsonl(public_path, public_sha256)
    device = torch.device("cuda")
    model, checkpoint = _load_world_model(
        args.ewc_checkpoint, args.ewc_checkpoint_sha256, device
    )
    cache = compile_typed_cache(
        model, public, device=device, batch_size=args.ewc_batch_size
    )
    model_state_sha256 = str(checkpoint["model_state_sha256"])
    del model
    torch.cuda.empty_cache()

    def learned_world(candidate: Mapping[str, Any]) -> TypedEpisode:
        try:
            return cache[str(candidate["episode_id"])]
        except KeyError as error:
            raise RuntimeError("EWC1/NPL2 WORLD cache miss") from error

    npl2.typed_episode_from_public = learned_world
    npl2.WORLD_OWNER_RECEIPT = f"learned-ewc1:{model_state_sha256}"
    npl2.WORLD_OWNER_CUSTODY = {
        "checkpoint": str(args.ewc_checkpoint),
        "checkpoint_sha256": args.ewc_checkpoint_sha256,
        "model_state_sha256": model_state_sha256,
        "qualification_result": str(args.ewc_result),
        "qualification_result_sha256": args.ewc_result_sha256,
        "source_deleted_after_compilation": "true",
    }
    npl2.DEVELOPMENT_SCHEMA = DEVELOPMENT_SCHEMA
    npl2.CONFIRMATION_SEED_SCHEMA = CONFIRMATION_SCHEMA
    sys.argv = [sys.argv[0], *remaining]
    npl2.main()


if __name__ == "__main__":
    main()
