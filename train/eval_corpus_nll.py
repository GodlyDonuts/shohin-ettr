#!/usr/bin/env python3
"""Measure pure next-token NLL on an immutable v3 holdout corpus."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import zstandard as zstd

from model import GPT, GPTConfig
from pipeline.tokenize_shards import (
    canonical_payload_sha256,
    file_receipt,
)
from pipeline.verify_tokenized_shards import verify_manifest


REPORT_SCHEMA = "shohin-fixed-corpus-nll-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CorpusNllError(ValueError):
    """The checkpoint or corpus cannot produce a comparable NLL receipt."""


def selected_window_indices(total_windows: int, selected_windows: int) -> list[int]:
    """Return deterministic midpoint-stratified indices over the full corpus."""
    if (
        total_windows < 1
        or selected_windows < 1
        or selected_windows > total_windows
    ):
        raise CorpusNllError("window-selection counts differ")
    indices = [
        ((2 * index + 1) * total_windows) // (2 * selected_windows)
        for index in range(selected_windows)
    ]
    if len(indices) != len(set(indices)) or indices[-1] >= total_windows:
        raise CorpusNllError("window selection is not unique and in range")
    return indices


def _iter_token_blocks(shard_paths: Iterable[Path]) -> Iterator[np.ndarray]:
    decompressor = zstd.ZstdDecompressor()
    pending = b""
    for path in shard_paths:
        try:
            with path.open("rb") as source:
                with decompressor.stream_reader(source) as reader:
                    while True:
                        block = reader.read(8 * 1024 * 1024)
                        if not block:
                            break
                        payload = pending + block
                        usable = len(payload) - (len(payload) % 2)
                        if usable:
                            yield np.frombuffer(
                                payload[:usable],
                                dtype=np.uint16,
                            )
                        pending = payload[usable:]
        except (OSError, zstd.ZstdError) as exc:
            raise CorpusNllError(f"corpus shard cannot be decoded: {path}") from exc
    if pending:
        raise CorpusNllError("corpus token stream has an odd byte count")


def iter_selected_windows(
    shard_paths: Iterable[Path],
    *,
    sequence_length: int,
    selected_indices: Iterable[int],
) -> Iterator[np.ndarray]:
    if sequence_length < 1:
        raise CorpusNllError("sequence length must be positive")
    targets = iter(selected_indices)
    try:
        target = next(targets)
    except StopIteration:
        return
    if target < 0:
        raise CorpusNllError("selected window index is negative")
    width = sequence_length + 1
    buffer = np.empty(0, dtype=np.uint16)
    window_index = 0
    for block in _iter_token_blocks(shard_paths):
        buffer = np.concatenate((buffer, block))
        offset = 0
        while len(buffer) - offset >= width:
            if window_index == target:
                yield buffer[offset : offset + width].copy()
                try:
                    following = next(targets)
                except StopIteration:
                    return
                if following <= target:
                    raise CorpusNllError(
                        "selected window indices are not increasing"
                    )
                target = following
            offset += width
            window_index += 1
        buffer = buffer[offset:].copy()
    raise CorpusNllError("selected window exceeds corpus token stream")


def _checkpoint_payload(
    checkpoint: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if HEX64.fullmatch(expected_sha256) is None:
        raise CorpusNllError("checkpoint SHA-256 is malformed")
    receipt = file_receipt(checkpoint)
    if receipt["sha256"] != expected_sha256:
        raise CorpusNllError("checkpoint SHA-256 differs")
    try:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError) as exc:
        raise CorpusNllError("checkpoint cannot be loaded safely") from exc
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("model"), Mapping)
        or not isinstance(payload.get("cfg"), Mapping)
    ):
        raise CorpusNllError("checkpoint model/config contract differs")
    return payload, receipt


def evaluate_corpus_nll(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    corpus_dir: Path,
    selection_code: Path,
    output: Path,
    max_target_tokens: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    if (
        max_target_tokens < 1
        or batch_size < 1
        or output.exists()
        or output.is_symlink()
        or not selection_code.is_file()
        or selection_code.is_symlink()
    ):
        raise CorpusNllError("NLL evaluation arguments differ")
    verification = verify_manifest(
        corpus_dir,
        selection_code=selection_code,
        require_external_inputs=True,
    )
    try:
        manifest = json.loads((corpus_dir / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusNllError("corpus manifest is unreadable") from exc
    payload, checkpoint_receipt = _checkpoint_payload(
        checkpoint,
        expected_sha256=checkpoint_sha256,
    )
    try:
        config = GPTConfig(**dict(payload["cfg"]))
    except (TypeError, ValueError) as exc:
        raise CorpusNllError("checkpoint GPT configuration differs") from exc
    if config.seq_len < 1:
        raise CorpusNllError("checkpoint sequence length differs")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise CorpusNllError("requested CUDA device is unavailable")
    model = GPT(config)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except RuntimeError as exc:
        raise CorpusNllError("checkpoint state differs from GPT configuration") from exc
    model.to(device)
    model.eval()
    torch.set_float32_matmul_precision("high")

    width = config.seq_len + 1
    total_windows = int(manifest["tokens"]) // width
    requested_windows = max_target_tokens // config.seq_len
    selected_windows = min(total_windows, max(1, requested_windows))
    selected_indices = selected_window_indices(total_windows, selected_windows)
    shard_paths = [
        corpus_dir / str(record["path"])
        for record in manifest["shard_files"]
    ]
    windows = iter_selected_windows(
        shard_paths,
        sequence_length=config.seq_len,
        selected_indices=selected_indices,
    )
    total_nll = 0.0
    evaluated_tokens = 0
    window_mean_nll: list[float] = []
    batches = 0
    started = time.monotonic()
    with torch.inference_mode():
        pending: list[np.ndarray] = []
        for window in windows:
            pending.append(window)
            if len(pending) < batch_size:
                continue
            total_nll, evaluated_tokens = _score_batch(
                model,
                pending,
                device=device,
                total_nll=total_nll,
                evaluated_tokens=evaluated_tokens,
                window_mean_nll=window_mean_nll,
            )
            pending = []
            batches += 1
        if pending:
            total_nll, evaluated_tokens = _score_batch(
                model,
                pending,
                device=device,
                total_nll=total_nll,
                evaluated_tokens=evaluated_tokens,
                window_mean_nll=window_mean_nll,
            )
            batches += 1
    expected_tokens = selected_windows * config.seq_len
    if (
        evaluated_tokens != expected_tokens
        or len(window_mean_nll) != selected_windows
        or not math.isfinite(total_nll)
        or total_nll <= 0
    ):
        raise CorpusNllError("NLL evaluation accounting differs")
    mean_nll = total_nll / evaluated_tokens
    elapsed = time.monotonic() - started
    report = {
        "schema": REPORT_SCHEMA,
        "checkpoint": checkpoint_receipt,
        "checkpoint_metadata": {
            "step": payload.get("step"),
            "data_seed": payload.get("data_seed"),
            "data_stream_generation": payload.get(
                "data_stream_generation"
            ),
            "data_contract": payload.get("data_contract"),
        },
        "corpus": {
            "path": str(corpus_dir.resolve()),
            "manifest_payload_sha256": manifest["payload_sha256"],
            "manifest_sha256": hashlib.sha256(
                (corpus_dir / "manifest.json").read_bytes()
            ).hexdigest(),
            "selection_code": file_receipt(selection_code),
            "verification": verification,
        },
        "sampling": {
            "algorithm": "midpoint_stratified_nonoverlapping_windows_v1",
            "sequence_length": config.seq_len,
            "corpus_tokens": manifest["tokens"],
            "total_complete_windows": total_windows,
            "selected_windows": selected_windows,
            "first_window_index": selected_indices[0],
            "last_window_index": selected_indices[-1],
            "target_tokens": evaluated_tokens,
        },
        "metric": {
            "name": "pure_next_token_cross_entropy",
            "training_zloss_excluded": True,
            "total_nll": total_nll,
            "mean_nll": mean_nll,
            "perplexity": math.exp(min(mean_nll, 80.0)),
            "window_mean_nll": window_mean_nll,
        },
        "runtime": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.startswith("cuda")
                else "cpu"
            ),
            "batch_size": batch_size,
            "batches": batches,
            "elapsed_seconds": elapsed,
            "target_tokens_per_second": evaluated_tokens / elapsed,
        },
    }
    report["payload_sha256"] = canonical_payload_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    return report


def _score_batch(
    model: GPT,
    windows: list[np.ndarray],
    *,
    device: str,
    total_nll: float,
    evaluated_tokens: int,
    window_mean_nll: list[float],
) -> tuple[float, int]:
    batch = torch.from_numpy(np.stack(windows).astype(np.int64)).to(device)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )
    with autocast_context:
        logits, _loss = model(inputs)
    token_nll = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view(targets.shape)
    sequence_nll = token_nll.sum(dim=1)
    window_mean_nll.extend(
        float(value) / targets.shape[1] for value in sequence_nll
    )
    return (
        total_nll + float(sequence_nll.sum()),
        evaluated_tokens + targets.numel(),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--selection-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-target-tokens", type=int, default=20_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    arguments = parser.parse_args(argv)
    result = evaluate_corpus_nll(
        checkpoint=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        corpus_dir=arguments.corpus_dir,
        selection_code=arguments.selection_code,
        output=arguments.output,
        max_target_tokens=arguments.max_target_tokens,
        batch_size=arguments.batch_size,
        device=arguments.device,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
