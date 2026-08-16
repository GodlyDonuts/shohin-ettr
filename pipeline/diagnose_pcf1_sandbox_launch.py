#!/usr/bin/env python3
"""Non-scientific stress probe for the PCF1 per-candidate launcher."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

from pcf1_code_sandbox import (
    PCF1SandboxError,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
    score_completion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--generate-before-score", action="store_true")
    return parser.parse_args()


def process_threads() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Threads:"):
            return int(line.split()[1])
    raise RuntimeError("Threads is absent from /proc/self/status")


def main() -> int:
    args = parse_args()
    if not os.environ.get("SLURM_TMPDIR"):
        raise RuntimeError("SLURM_TMPDIR is required")
    selected = None
    with args.data.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == args.row:
                selected = json.loads(line)
                break
    if selected is None or selected.get("task") != "mbpp":
        raise RuntimeError("selected row is not the expected MBPP row")
    assessor = selected["assessor"]
    sleepers: list[threading.Thread] = []
    stop = threading.Event()
    for _ in range(args.threads):
        thread = threading.Thread(target=stop.wait, daemon=True)
        thread.start()
        sleepers.append(thread)
    allocation_receipt = qualify_allocation()
    receipts = qualify_mbpp_assessor_setups([assessor])
    model = None
    model_receipt = None
    if (args.model_root is None) != (args.adapter_checkpoint is None):
        raise RuntimeError("model root and adapter checkpoint must be paired")
    if args.generate_before_score and args.model_root is None:
        raise RuntimeError("generation stress requires the model and adapter")
    tokenizer = None
    generation = None
    if args.model_root is not None:
        import torch
        from transformers import AutoTokenizer

        from hf_pcf1_evaluate import (
            _generate_completions,
            _generation_stop_token_ids,
            _load_model,
            _render_prompt,
            validate_adapter_trainables,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_root,
            local_files_only=True,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model, adapter_metadata, loader = _load_model(
            args.model_root, args.adapter_checkpoint, "multimodal"
        )
        if loader != "multimodal":
            raise RuntimeError("diagnostic model loader differs")
        model_receipt = validate_adapter_trainables(model, adapter_metadata)
        torch.cuda.synchronize()
        if args.generate_before_score:
            rendered = _render_prompt(tokenizer, str(selected["question"]), True, False)
            stop_ids = _generation_stop_token_ids(tokenizer)

            def generation() -> str:
                completions, _ = _generate_completions(
                    model,
                    tokenizer,
                    [rendered],
                    True,
                    "greedy",
                    768,
                    stop_ids,
                )
                return completions[0]

    print(
        json.dumps(
            {
                "event": "start",
                "identity_sha256": selected["identity_sha256"],
                "iterations": args.iterations,
                "allocation_probe_sha256": allocation_receipt["probe_sha256"],
                "requested_background_threads": args.threads,
                "process_threads": process_threads(),
                "setup_receipts": len(receipts),
                "model_loaded": model is not None,
                "generate_before_score": args.generate_before_score,
                "model_receipt": model_receipt,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    completion = "def pcf1_diagnostic_placeholder(*args, **kwargs):\n    return None\n"
    started = time.monotonic()
    try:
        for iteration in range(1, args.iterations + 1):
            call_started = time.monotonic()
            candidate = generation() if generation is not None else completion
            try:
                result = score_completion(assessor, candidate)
            except PCF1SandboxError as error:
                print(
                    json.dumps(
                        {
                            "event": "infrastructure_failure",
                            "iteration": iteration,
                            "message": str(error),
                            "process_threads": process_threads(),
                            "elapsed_seconds": time.monotonic() - call_started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise
            if generation is None and result.get("correct") is not False:
                raise RuntimeError("diagnostic placeholder unexpectedly passed")
            if iteration % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "iteration": iteration,
                            "process_threads": process_threads(),
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        stop.set()
        for thread in sleepers:
            thread.join(timeout=1)
    print(
        json.dumps(
            {
                "event": "complete",
                "iterations": args.iterations,
                "elapsed_seconds": time.monotonic() - started,
                "process_threads": process_threads(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
