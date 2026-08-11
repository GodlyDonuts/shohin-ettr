"""Exact PCF1 allocated-node environment receipt verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "shohin-pcf1-environment-receipt-v1"
ENVIRONMENT_ROOT = "/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2"
ENVIRONMENT_MANIFEST_SHA256 = (
    "7eec9e1e94da3480820458912cb96f4ee13c3427543b3addb5ea31953b5d1971"
)
ENVIRONMENT_RUNTIME_SHA256 = (
    "277b97fbd6b18760c9789cf3f3372bdb6b40ca87bf84a1df4b41ee3194c4e9dd"
)
PIP_FREEZE_SHA256 = "1d4dfd4a1dc11af9788b0bab072d262278db1814d3fca49465d4df5931b3b87a"
ENVIRONMENT_TREE = {
    "sha256": "6c3311032bc4efb065222378e053e1cc15266b37bd868aee2bc05aa94f8ebf9c",
    "entries": 9_416,
    "files": 8_022,
    "directories": 1_390,
    "symlinks": 4,
    "file_bytes": 170_408_093,
}
PYTHON = {
    "implementation": "CPython",
    "version": "3.13.13",
    "entrypoint": f"{ENVIRONMENT_ROOT}/bin/python",
    "resolved_executable": (
        "/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13"
    ),
    "executable_sha256": (
        "051a031d827eab9778e982571db754662809164c8a3ec01e9beea1e1088123e0"
    ),
    "prefix": ENVIRONMENT_ROOT,
    "base_prefix": "/lustre/fs1/home/sa305415/shohin/miniforge3",
    "site_packages": f"{ENVIRONMENT_ROOT}/lib/python3.13/site-packages",
}
PACKAGE_PINS = {
    "accelerate": "1.14.0",
    "huggingface-hub": "1.22.0",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
    "tokenizers": "0.22.2",
    "torch": "2.6.0+cu124",
    "transformers": "5.15.0.dev0",
    "triton": "3.2.0",
}
AUTO_TOKENIZER_CLASS = "transformers.models.auto.tokenization_auto.AutoTokenizer"
MULTIMODAL_AUTO_CLASS = (
    "transformers.models.auto.modeling_auto.AutoModelForMultimodalLM"
)
MODULE_ORIGINS = {
    "accelerate": f"{ENVIRONMENT_ROOT}/lib/python3.13/site-packages/accelerate/__init__.py",
    "huggingface-hub": "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/huggingface_hub/__init__.py",
    "peft": f"{ENVIRONMENT_ROOT}/lib/python3.13/site-packages/peft/__init__.py",
    "safetensors": f"{ENVIRONMENT_ROOT}/lib/python3.13/site-packages/safetensors/__init__.py",
    "sentencepiece": f"{ENVIRONMENT_ROOT}/lib/python3.13/site-packages/sentencepiece/__init__.py",
    "tokenizers": f"{ENVIRONMENT_ROOT}/lib/python3.13/site-packages/tokenizers/__init__.py",
    "torch": "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch/__init__.py",
    "transformers": f"{ENVIRONMENT_ROOT}/lib/python3.13/site-packages/transformers/__init__.py",
    "triton": "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/triton/__init__.py",
}
RUNTIME_SOURCE_DEPENDENCIES = frozenset(
    {
        "pipeline/build_pcf1_custody.py",
        "pipeline/capture_pcf1_environment.py",
        "pipeline/normalize_pcf1_reports.py",
        "pipeline/package_pcf1_runtime.py",
        "pipeline/score_pcf1_commit.py",
        "train/hf_pcf1_apply_commit.py",
        "train/hf_pcf1_evaluate.py",
        "train/hf_pcf1_generate_drafts.py",
        "train/hf_pcf1_mechanics.py",
        "train/hf_pcf1_train_commit.py",
        "train/hf_product_reasoning_eval.py",
        "train/hf_product_reasoning_train.py",
        "train/pcf1_code_sandbox.py",
        "train/pcf1_environment.py",
        "train/jobs/pcf1_common.sh",
    }
)


class PCF1EnvironmentError(RuntimeError):
    """The explicit receipt differs from the qualified PCF1 runtime."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_environment_receipt(
    path: Path, expected_sha256: str, required_source: str
) -> dict[str, Any]:
    """Validate explicit bytes, frozen host identity, and the calling source hash."""

    rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
    if any(term in rendered for term in ("holdout", "product", "public")):
        raise PCF1EnvironmentError("protected path supplied as environment receipt")
    if (
        not _sha256(expected_sha256)
        or path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != expected_sha256
    ):
        raise PCF1EnvironmentError("PCF1 environment receipt bytes differ")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PCF1EnvironmentError("PCF1 environment receipt is unreadable") from error
    packages = receipt.get("packages") if isinstance(receipt, dict) else None
    sources = (
        receipt.get("runtime_source_sha256s") if isinstance(receipt, dict) else None
    )
    expected_keys = {
        "schema",
        "status",
        "python",
        "packages",
        "module_origins",
        "torch_version",
        "transformers_version",
        "tokenizers_version",
        "cuda_build_version",
        "cuda_compiled_version",
        "auto_tokenizer_class",
        "multimodal_auto_class",
        "environment_root",
        "environment_manifest_sha256",
        "environment_manifest_entries",
        "environment_tree",
        "environment_runtime_sha256",
        "pip_freeze_sha256",
        "runtime_root",
        "runtime_manifest_sha256",
        "runtime_source_sha256s",
        "offline_required",
        "bytecode_writes_permitted",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_keys
        or receipt.get("schema") != SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("environment_root") != ENVIRONMENT_ROOT
        or receipt.get("environment_manifest_sha256") != ENVIRONMENT_MANIFEST_SHA256
        or receipt.get("environment_manifest_entries") != 2
        or receipt.get("environment_runtime_sha256") != ENVIRONMENT_RUNTIME_SHA256
        or receipt.get("pip_freeze_sha256") != PIP_FREEZE_SHA256
        or receipt.get("environment_tree") != ENVIRONMENT_TREE
        or receipt.get("python") != PYTHON
        or receipt.get("module_origins") != MODULE_ORIGINS
        or receipt.get("torch_version") != "2.6.0+cu124"
        or receipt.get("transformers_version") != "5.15.0.dev0"
        or receipt.get("tokenizers_version") != "0.22.2"
        or receipt.get("cuda_build_version") != "12.4"
        or receipt.get("cuda_compiled_version") != 12040
        or receipt.get("auto_tokenizer_class") != AUTO_TOKENIZER_CLASS
        or receipt.get("multimodal_auto_class") != MULTIMODAL_AUTO_CLASS
        or receipt.get("offline_required") is not True
        or receipt.get("bytecode_writes_permitted") is not False
        or not _sha256(receipt.get("runtime_manifest_sha256"))
        or not isinstance(receipt.get("runtime_root"), str)
        or packages != PACKAGE_PINS
        or not isinstance(sources, dict)
        or set(sources) != RUNTIME_SOURCE_DEPENDENCIES
        or any(not _sha256(value) for value in sources.values())
        or not _sha256(sources.get(required_source))
    ):
        raise PCF1EnvironmentError("PCF1 environment receipt content differs")
    runtime_root = Path(receipt["runtime_root"])
    source = runtime_root / required_source
    if (
        runtime_root.is_symlink()
        or not runtime_root.is_dir()
        or source.is_symlink()
        or not source.is_file()
        or sha256_file(source) != sources[required_source]
    ):
        raise PCF1EnvironmentError("PCF1 calling runtime source differs")
    return receipt
