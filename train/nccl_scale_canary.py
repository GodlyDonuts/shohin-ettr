#!/usr/bin/env python3
"""Bounded NCCL all-reduce canary for a pre-existing Slurm allocation.

This does not load a model, optimizer, or corpus. It only verifies that the
allocated GPU topology can sustain a deterministic multi-node collective
before a data-bound distributed training pilot is submitted.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def env_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"missing required Slurm environment variable: {name}")
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--master-port", type=int, required=True)
    args = parser.parse_args()
    if args.elements <= 0 or args.warmup < 0 or args.steps <= 0:
        raise ValueError("elements and steps must be positive; warmup must be nonnegative")

    rank = env_int("SLURM_PROCID")
    world_size = env_int("SLURM_NTASKS")
    local_rank = env_int("SLURM_LOCALID")
    master_addr = os.environ.get("MASTER_ADDR")
    if not master_addr:
        raise RuntimeError("MASTER_ADDR must name the rank-zero node")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{args.master_port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # A small exact collective catches rank/topology mistakes before timing.
        check = torch.tensor([rank + 1], dtype=torch.float32, device="cuda")
        dist.all_reduce(check)
        expected = world_size * (world_size + 1) / 2
        if check.item() != expected:
            raise RuntimeError(f"collective correctness mismatch: {check.item()} != {expected}")

        payload = torch.ones(args.elements, dtype=torch.bfloat16, device="cuda")
        for _ in range(args.warmup):
            dist.all_reduce(payload)
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        for _ in range(args.steps):
            dist.all_reduce(payload)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        maximum = torch.tensor([elapsed], dtype=torch.float64, device="cuda")
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if rank == 0:
            total_bytes = args.elements * torch.empty((), dtype=torch.bfloat16).element_size()
            seconds_per_collective = maximum.item() / args.steps
            print(
                json.dumps(
                    {
                        "schema": "shohin-nccl-scale-canary-v1",
                        "world_size": world_size,
                        "elements": args.elements,
                        "payload_bytes": total_bytes,
                        "warmup": args.warmup,
                        "steps": args.steps,
                        "max_seconds": maximum.item(),
                        "seconds_per_all_reduce": seconds_per_collective,
                        "algorithmic_payload_gb_per_second": total_bytes
                        / seconds_per_collective
                        / 1e9,
                        "gpu": torch.cuda.get_device_name(local_rank),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
