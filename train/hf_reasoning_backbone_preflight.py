"""Download and smoke-test a pinned Hugging Face reasoning backbone.

The preflight runs on an allocated GPU node because Newton login nodes cannot
reliably import the numerical stack.  It records enough environment and
generation information to decide whether a backbone is ready for the product
benchmark campaign; it never reads or emits provider credentials.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any


class BackbonePreflightError(RuntimeError):
    """The pinned backbone could not satisfy the runtime contract."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_model(model_root: Path):
    import torch

    common = {
        "pretrained_model_name_or_path": str(model_root),
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }
    attempts: list[dict[str, str]] = []
    try:
        from transformers import AutoModelForMultimodalLM

        return AutoModelForMultimodalLM.from_pretrained(**common), attempts
    except Exception as exc:  # pragma: no cover - depends on installed Transformers
        attempts.append(
            {
                "loader": "AutoModelForMultimodalLM",
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
        )

    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(**common), attempts
    except Exception as exc:  # pragma: no cover - depends on installed Transformers
        attempts.append(
            {
                "loader": "AutoModelForCausalLM",
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
        )

    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(**common), attempts
    except Exception as exc:  # pragma: no cover - depends on installed Transformers
        attempts.append(
            {
                "loader": "AutoModelForImageTextToText",
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
        )
    raise BackbonePreflightError(json.dumps(attempts, sort_keys=True))


def _render_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    return f"User: {prompt}\n\nAssistant:"


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    if not torch.cuda.is_available():
        raise BackbonePreflightError("CUDA is unavailable")

    started = time.monotonic()
    model_root = Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            local_dir=args.model_root,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.txt",
                "*.tiktoken",
                "*.py",
            ],
        )
    ).resolve()
    download_seconds = time.monotonic() - started

    config = AutoConfig.from_pretrained(model_root, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_root, trust_remote_code=True)
    model, failed_loaders = _load_model(model_root)
    model.eval()

    rendered = _render_prompt(tokenizer, args.prompt)
    encoded = tokenizer(rendered, return_tensors="pt")
    encoded = {key: value.to("cuda:0") for key, value in encoded.items()}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    generation_started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    generation_seconds = time.monotonic() - generation_started
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    new_tokens = int(generated.shape[-1] - prompt_tokens)
    completion = tokenizer.decode(
        generated[0, prompt_tokens:], skip_special_tokens=True
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    device = torch.cuda.get_device_properties(0)
    return {
        "schema": "shohin-hf-reasoning-backbone-preflight-v1",
        "status": "pass",
        "model": args.model,
        "revision": args.revision,
        "model_root": str(model_root),
        "model_class": type(model).__name__,
        "config_class": type(config).__name__,
        "model_type": getattr(config, "model_type", None),
        "parameter_count": parameter_count,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "dtype": str(next(model.parameters()).dtype),
        "prompt": args.prompt,
        "thinking_mode_requested": True,
        "rendered_prompt": rendered,
        "completion": completion,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": new_tokens,
        "generation_seconds": generation_seconds,
        "generated_tokens_per_second": (
            new_tokens / generation_seconds if generation_seconds else None
        ),
        "download_seconds": download_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": {
            "name": device.name,
            "total_memory_bytes": int(device.total_memory),
            "compute_capability": f"{device.major}.{device.minor}",
        },
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": _package_version("huggingface-hub"),
            "accelerate": _package_version("accelerate"),
            "peft": _package_version("peft"),
            "lm_eval": _package_version("lm-eval"),
        },
        "failed_loaders": failed_loaders,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt",
        default=(
            "A box contains 3 red balls and 5 blue balls. Two blue balls are "
            "removed, then the number of red balls is doubled. How many balls "
            "are now in the box? Explain briefly, then give the final integer."
        ),
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = run_preflight(args)
    except Exception as exc:
        report = {
            "schema": "shohin-hf-reasoning-backbone-preflight-v1",
            "status": "fail",
            "model": args.model,
            "revision": args.revision,
            "error": f"{type(exc).__name__}: {exc}"[:12000],
        }
        _atomic_json(args.output, report)
        raise
    _atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
