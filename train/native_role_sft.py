#!/usr/bin/env python3
"""Train one native Shohin draft/revision role while freezing the shared trunk."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Iterator

import numpy as np
import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from native_role_lora import (
    NativeRoleLoRAConfig,
    attach_role_lora,
    export_role_adapter,
    load_role_adapter,
    role_parameter_count,
    set_trainable_role,
)
from sft import DEFAULT_Q_FIELDS, DEFAULT_R_FIELDS, build_packed, sha256_file


TRAINING_REPORT_SCHEMA = "shohin-native-role-sft-v1"


def verify_sha256(path: str | Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if expected and actual != expected:
        raise ValueError(f"{label} SHA-256 differs: expected {expected}, got {actual}")
    return actual


def deterministic_batch_indices(
    count: int, batch_size: int, updates: int, seed: int
) -> Iterator[np.ndarray]:
    """Yield full seeded batches, starting a fresh permutation when exhausted."""
    if count < batch_size:
        raise ValueError("packed sequence count is smaller than batch size")
    if batch_size <= 0 or updates <= 0:
        raise ValueError("batch size and update count must be positive")
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    cursor = 0
    for _ in range(updates):
        if cursor + batch_size > count:
            order = rng.permutation(count)
            cursor = 0
        result = order[cursor : cursor + batch_size]
        cursor += batch_size
        yield result


def cosine_multiplier(step: int, total: int, warmup: int) -> float:
    if total <= 0 or warmup < 0:
        raise ValueError("schedule geometry is invalid")
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as sink:
            torch.save(payload, sink)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    try:
        with temporary.open("xb") as sink:
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_adapter_payload(path: str | Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"native role adapter payload is invalid: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--role", choices=("draft", "revision"), required=True)
    parser.add_argument("--draft-adapter", default="")
    parser.add_argument("--revision-adapter", default="")
    parser.add_argument("--adapter-layers", type=int, default=4)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--adapter-alpha", type=float, default=16.0)
    parser.add_argument("--pack-len", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--exact-updates", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--q-fields", nargs="+", default=DEFAULT_Q_FIELDS)
    parser.add_argument("--r-fields", nargs="+", default=DEFAULT_R_FIELDS)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--expected-init-sha256", default="")
    parser.add_argument("--expected-tokenizer-sha256", default="")
    parser.add_argument(
        "--expected-data-sha256",
        action="append",
        default=[],
        metavar="SHA256",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("epochs and batch size must be positive")
    if args.exact_updates < 0 or args.learning_rate <= 0 or args.clip <= 0:
        raise SystemExit("update count, learning rate, and clip are invalid")
    if args.warmup < 0 or args.log_every <= 0:
        raise SystemExit("warmup and logging cadence are invalid")
    if args.expected_data_sha256 and len(args.expected_data_sha256) != len(args.data):
        raise SystemExit("--expected-data-sha256 must match --data one-for-one")

    output = Path(args.out)
    staging = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if output.exists() or output.is_symlink() or staging.exists():
        raise FileExistsError(f"native role output already exists: {output}")
    staging.mkdir(parents=True, mode=0o700)

    try:
        init_sha256 = verify_sha256(args.init, args.expected_init_sha256, "init")
        tokenizer_sha256 = verify_sha256(
            args.tokenizer, args.expected_tokenizer_sha256, "tokenizer"
        )
        expected_data = args.expected_data_sha256 or [""] * len(args.data)
        data_sha256 = {
            path: verify_sha256(path, expected, f"data[{index}]")
            for index, (path, expected) in enumerate(zip(args.data, expected_data))
        }

        checkpoint = torch.load(args.init, map_location="cpu")
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("cfg"), dict):
            raise ValueError("init checkpoint lacks an exact model config")
        cfg = GPTConfig(**checkpoint["cfg"])
        pack_len = args.pack_len or cfg.seq_len
        if pack_len <= 1 or pack_len > cfg.seq_len:
            raise ValueError("pack length is outside the model context")

        tokenizer = Tokenizer.from_file(args.tokenizer)
        eos_id = tokenizer.token_to_id("<|endoftext|>")
        if eos_id is None:
            raise ValueError("tokenizer does not define <|endoftext|>")
        x_data, y_data, _ = build_packed(
            args.data,
            tokenizer,
            pack_len,
            args.q_fields,
            args.r_fields,
            eos_id,
            args.max_examples,
        )
        packed_count = len(x_data)
        if packed_count < args.batch_size:
            raise ValueError("packed sequence count is smaller than batch size")
        updates = args.exact_updates or args.epochs * (packed_count // args.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

        model = GPT(cfg).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        adapter_config = NativeRoleLoRAConfig(
            layers=args.adapter_layers,
            rank=args.adapter_rank,
            alpha=args.adapter_alpha,
        )
        attach_role_lora(model, adapter_config)
        loaded_adapters = {}
        for role, path in (
            ("draft", args.draft_adapter),
            ("revision", args.revision_adapter),
        ):
            if not path:
                continue
            payload = load_adapter_payload(path)
            load_role_adapter(
                model,
                payload,
                adapter_config,
                expected_role=role,
                expected_base_checkpoint_sha256=init_sha256,
            )
            loaded_adapters[role] = {
                "path": path,
                "sha256": sha256_file(path),
            }

        trainable_count = set_trainable_role(model, args.role)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if sum(parameter.numel() for parameter in trainable) != trainable_count:
            raise RuntimeError("trainable role parameter inventory differs")
        if trainable_count != role_parameter_count(model, args.role):
            raise RuntimeError("role parameter count receipt differs")

        raw_model = model.train()
        execution_model = torch.compile(model) if args.compile else model
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        losses = []
        supervised_tokens = 0
        started = time.time()
        for step, indices in enumerate(
            deterministic_batch_indices(
                packed_count, args.batch_size, updates, args.seed
            )
        ):
            x = torch.from_numpy(x_data[indices]).to(device)
            y = torch.from_numpy(y_data[indices]).to(device)
            supervised_tokens += int(np.count_nonzero(y_data[indices] != -1))
            multiplier = cosine_multiplier(step, updates, args.warmup)
            optimizer.param_groups[0]["lr"] = args.learning_rate * multiplier
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                _, loss = execution_model(x, y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at update {step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError(f"non-finite gradient norm at update {step}")
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % args.log_every == 0 or step + 1 == updates:
                elapsed = time.time() - started
                print(
                    f"[native-role-sft] role={args.role} update={step + 1}/{updates} "
                    f"loss={losses[-1]:.4f} lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        adapter_payload = export_role_adapter(
            raw_model,
            adapter_config,
            args.role,
            base_checkpoint_sha256=init_sha256,
        )
        adapter_path = staging / f"{args.role}_adapter.pt"
        atomic_torch_save(adapter_path, adapter_payload)
        adapter_sha256 = sha256_file(adapter_path)
        duration = time.time() - started
        report = {
            "schema": TRAINING_REPORT_SCHEMA,
            "status": "complete",
            "role": args.role,
            "base": {
                "path": args.init,
                "sha256": init_sha256,
                "step": checkpoint.get("step"),
                "config": checkpoint["cfg"],
            },
            "adapter": {
                "config": adapter_payload["adapter_config"],
                "path": adapter_path.name,
                "sha256": adapter_sha256,
                "trainable_parameters": trainable_count,
            },
            "loaded_adapters": loaded_adapters,
            "data": {
                "paths": args.data,
                "sha256": data_sha256,
                "tokenizer_path": args.tokenizer,
                "tokenizer_sha256": tokenizer_sha256,
                "pack_len": pack_len,
                "packed_sequences": packed_count,
                "supervised_tokens_consumed": supervised_tokens,
            },
            "optimization": {
                "optimizer": "AdamW",
                "betas": [0.9, 0.95],
                "weight_decay": 0.0,
                "learning_rate": args.learning_rate,
                "schedule": "warmup_cosine_to_0.1x",
                "warmup": args.warmup,
                "clip": args.clip,
                "updates": updates,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "compile": args.compile,
            },
            "result": {
                "first_loss": losses[0],
                "final_loss": losses[-1],
                "minimum_loss": min(losses),
                "mean_loss": sum(losses) / len(losses),
                "duration_seconds": duration,
                "updates_per_minute": updates / max(duration, 1e-9) * 60.0,
            },
            "runtime": {
                "device": str(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        }
        atomic_json(staging / "training_report.json", report)
        os.replace(staging, output)
        print(
            f"[native-role-sft] complete role={args.role} adapter={adapter_sha256} "
            f"output={output}",
            flush=True,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
