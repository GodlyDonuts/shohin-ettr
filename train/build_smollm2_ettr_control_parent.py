#!/usr/bin/env python3
"""Build an isolated SmolLM2 + randomly initialized ETTR control parent."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Sequence

import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
)
from ettr_token_transcode import (
    TokenNativeETTRTranscoder,
    receipt_value as transcode_receipt_value,
)
from smollm2_backbone import (
    SmolLM2BackboneReceipt,
    file_sha256,
    import_smollm2_135m,
)
from train_ettr_joint_instruction_canary import MODEL_SCHEMA, RUN_SCHEMA
from train_ettr_joint_stream_canary import (
    _canonical_bytes,
    _torch_save_no_replace,
    _write_no_replace,
)


REPORT_SCHEMA = "shohin-smollm2-ettr-control-parent-report-v1"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class SmolLM2ETTRParentError(RuntimeError):
    """The isolated cross-backbone parent contract differs."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-tokenizer", type=Path, required=True)
    parser.add_argument("--target-tokenizer", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--architecture-seed", type=int, required=True)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if (
        not args.source_root.is_absolute()
        or not args.source_root.is_dir()
        or not args.source_tokenizer.is_absolute()
        or not args.source_tokenizer.is_file()
        or not args.target_tokenizer.is_absolute()
        or not args.target_tokenizer.is_file()
        or args.source_tokenizer == args.target_tokenizer
        or _HEX64.fullmatch(args.release_sha256) is None
        or _HEX40.fullmatch(args.source_commit) is None
        or not isinstance(args.architecture_seed, int)
        or isinstance(args.architecture_seed, bool)
        or not 0 <= args.architecture_seed < 2**63
        or not args.output.is_absolute()
        or not args.output.parent.is_dir()
        or args.output.exists()
        or args.output.is_symlink()
    ):
        raise SmolLM2ETTRParentError(
            "SmolLM2 ETTR parent arguments differ"
        )


def _import_value(receipt: SmolLM2BackboneReceipt) -> dict[str, object]:
    return {
        name: getattr(receipt, name)
        for name in receipt.__dataclass_fields__
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    target_tokenizer_sha256 = file_sha256(args.target_tokenizer)
    base, base_receipt = import_smollm2_135m(
        args.source_root,
        tokenizer_sha256=target_tokenizer_sha256,
        dtype=torch.bfloat16,
    )
    transcoder = TokenNativeETTRTranscoder(
        args.source_tokenizer,
        args.target_tokenizer,
    )
    if transcoder.target_vocab_size != base.cfg.vocab_size:
        raise SmolLM2ETTRParentError(
            "SmolLM2 tokenizer vocabulary differs"
        )
    config = TheoryReactorConfig()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.architecture_seed)
        model = EndogenousTypedTheoryReactorGPT(base, config)
    model.to(dtype=torch.bfloat16)
    model.eval()
    base_parameters = base.num_params()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    architecture_parameters = total_parameters - base_parameters
    initialization = {
        "architecture_seed": args.architecture_seed,
        "base_import": _import_value(base_receipt),
        "initialization": "external-smollm2-135m-control",
        "token_transcode": transcode_receipt_value(transcoder.receipt),
    }
    run_contract = {
        "architecture_seed": args.architecture_seed,
        "ettr_release_sha256": args.release_sha256,
        "initialization": initialization,
        "model_config": asdict(config),
        "schema": RUN_SCHEMA,
        "source_commit": args.source_commit,
        "token_transcode": transcode_receipt_value(transcoder.receipt),
    }
    run_contract_bytes = _canonical_bytes(run_contract)
    run_contract_sha256 = hashlib.sha256(run_contract_bytes).hexdigest()
    payload = {
        "base_config": asdict(base.cfg),
        "base_import": _import_value(base_receipt),
        "base_rms_norm_eps": base_receipt.rms_norm_eps,
        "ettr_config": asdict(config),
        "initialization": initialization,
        "model": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        },
        "optimizer_step": 0,
        "run_contract_sha256": run_contract_sha256,
        "schedule": {
            "optimizer_step": 0,
            "schema": "shohin-smollm2-ettr-control-schedule-v1",
        },
        "schema": MODEL_SCHEMA,
        "source_commit": args.source_commit,
    }
    try:
        args.output.mkdir(mode=0o700)
        observed_contract_sha256 = _write_no_replace(
            args.output / "run-contract.json",
            run_contract_bytes,
        )
        if observed_contract_sha256 != run_contract_sha256:
            raise SmolLM2ETTRParentError(
                "SmolLM2 run-contract hash differs"
            )
        model_sha256 = _torch_save_no_replace(
            args.output / "joint-model-final.pt",
            payload,
        )
        report = {
            "architecture_parameters": architecture_parameters,
            "base_import": _import_value(base_receipt),
            "base_parameters": base_parameters,
            "experimental_over_shohin_cap": max(
                0,
                total_parameters - 200_000_000,
            ),
            "joint_model_sha256": model_sha256,
            "run_contract_sha256": run_contract_sha256,
            "schema": REPORT_SCHEMA,
            "source_commit": args.source_commit,
            "token_transcode": transcode_receipt_value(
                transcoder.receipt
            ),
            "total_parameters": total_parameters,
        }
        _write_no_replace(
            args.output / "parent-report.json",
            _canonical_bytes(report),
        )
        checksums = []
        for name in (
            "joint-model-final.pt",
            "parent-report.json",
            "run-contract.json",
        ):
            checksums.append(
                f"{file_sha256(args.output / name)}  {name}\n"
            )
        _write_no_replace(
            args.output / "SHA256SUMS",
            "".join(checksums).encode("ascii"),
        )
        for path in args.output.iterdir():
            os.chmod(path, 0o400)
        os.chmod(args.output, 0o500)
    except BaseException:
        shutil.rmtree(args.output, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "joint_model_sha256": model_sha256,
                "output": str(args.output),
                "run_contract_sha256": run_contract_sha256,
                "total_parameters": total_parameters,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
