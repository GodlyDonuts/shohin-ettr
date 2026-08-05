#!/usr/bin/env python3
"""Audit fieldwise errors from a frozen DIVERGE component-pilot checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from diverge_v0_neural_pilot import (
    SmolDivergePilotCompiler,
    generate_episode,
    predict_episode,
)
from frozen_pointer_backbone import load_frozen_pointer_backbone


SCHEMA = "shohin-diverge-v0-neural-error-audit-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _counter(counter: Counter[tuple[object, ...]]) -> dict[str, int]:
    return {"|".join(map(str, key)): value for key, value in sorted(counter.items())}


def _audit_split(
    model: SmolDivergePilotCompiler,
    *,
    split: str,
    count: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    width, renderer, ontology = {
        "development": (5, 2, "parcel-relation"),
        "confirmation": (6, 3, "signal-routing"),
    }[split]
    kind = Counter()
    program = Counter()
    prior = Counter()
    primary_program = Counter()
    primary_prior = Counter()
    examples = []
    for index in range(count):
        episode = generate_episode(
            seed=seed + index,
            split=split,
            width=width,
            renderer=renderer,
            ontology=ontology,
        )
        prediction = predict_episode(model, episode, device)
        for record_index, record in enumerate(episode.records):
            kind[(int(record.is_fault_line), int(prediction.selected[record_index]))] += 1
            if not record.is_fault_line:
                continue
            primary = record.record_id == episode.primary_record_id
            for option_index, option in enumerate(record.options):
                program_key = (option.program, prediction.programs[record_index][option_index])
                prior_key = (option.prior_class, prediction.priors[record_index][option_index])
                program[program_key] += 1
                prior[prior_key] += 1
                if primary:
                    primary_program[program_key] += 1
                    primary_prior[prior_key] += 1
                    if (
                        len(examples) < 24
                        and (program_key[0] != program_key[1] or prior_key[0] != prior_key[1])
                    ):
                        examples.append(
                            {
                                "episode": episode.episode_id,
                                "option_text": option.text,
                                "true_program": option.program,
                                "predicted_program": prediction.programs[record_index][option_index],
                                "true_prior": option.prior_class,
                                "predicted_prior": prediction.priors[record_index][option_index],
                            }
                        )
    return {
        "split": split,
        "count": count,
        "width": width,
        "renderer": renderer,
        "ontology": ontology,
        "kind_confusion_true_pred": _counter(kind),
        "program_confusion_true_pred": _counter(program),
        "prior_confusion_true_pred": _counter(prior),
        "primary_program_confusion_true_pred": _counter(primary_program),
        "primary_prior_confusion_true_pred": _counter(primary_prior),
        "primary_error_examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--eval-seed", type=int, default=202608053000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    arguments = payload["arguments"]
    backbone, _, receipt = load_frozen_pointer_backbone(args.base, device=device)
    model = SmolDivergePilotCompiler(
        backbone,
        Tokenizer.from_file(str(args.tokenizer)),
        layer=int(arguments["layer"]),
        width=int(arguments["width"]),
        char_width=int(arguments["char_width"]),
        selection_policy=str(arguments["selection_policy"]),
    ).to(device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    nonbackbone_missing = [name for name in missing if not name.startswith("backbone.")]
    if nonbackbone_missing or unexpected:
        raise ValueError(
            f"adapter mismatch missing={nonbackbone_missing} unexpected={list(unexpected)}"
        )
    model.eval()
    report = {
        "schema": SCHEMA,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "base_sha256": _sha256(args.base),
        "tokenizer_sha256": _sha256(args.tokenizer),
        "checkpoint_format": receipt.checkpoint_format,
        "evaluations": [
            _audit_split(
                model,
                split=split,
                count=args.count,
                seed=args.eval_seed + offset,
                device=device,
            )
            for split, offset in (("development", 0), ("confirmation", 100_000))
        ],
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    _atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
