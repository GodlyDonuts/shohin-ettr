"""Focused exact-closure tests for the PCF1 environment receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import pcf1_environment as environment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> tuple[Path, str, str]:
    runtime = root / "runtime"
    source_hashes: dict[str, str] = {}
    for relative in sorted(environment.RUNTIME_SOURCE_DEPENDENCIES):
        path = runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
        source_hashes[relative] = _sha256(path)
    required_source = "train/pcf1_environment.py"
    receipt: dict[str, Any] = {
        "schema": environment.SCHEMA,
        "status": "complete",
        "python": dict(environment.PYTHON),
        "packages": dict(environment.PACKAGE_PINS),
        "module_origins": dict(environment.MODULE_ORIGINS),
        "torch_version": "2.6.0+cu124",
        "transformers_version": "5.15.0.dev0",
        "tokenizers_version": "0.22.2",
        "cuda_build_version": "12.4",
        "cuda_compiled_version": 12040,
        "auto_tokenizer_class": environment.AUTO_TOKENIZER_CLASS,
        "multimodal_auto_class": environment.MULTIMODAL_AUTO_CLASS,
        "environment_root": environment.ENVIRONMENT_ROOT,
        "environment_manifest_sha256": environment.ENVIRONMENT_MANIFEST_SHA256,
        "environment_manifest_entries": 2,
        "environment_tree": dict(environment.ENVIRONMENT_TREE),
        "environment_runtime_sha256": environment.ENVIRONMENT_RUNTIME_SHA256,
        "pip_freeze_sha256": environment.PIP_FREEZE_SHA256,
        "runtime_root": str(runtime),
        "runtime_manifest_sha256": "a" * 64,
        "runtime_source_sha256s": source_hashes,
        "offline_required": True,
        "bytecode_writes_permitted": False,
    }
    path = root / "environment.json"
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return path, _sha256(path), required_source


def test_accepts_exact_closed_environment_receipt(tmp_path: Path) -> None:
    path, digest, source = _fixture(tmp_path)

    receipt = environment.validate_environment_receipt(path, digest, source)

    assert receipt["auto_tokenizer_class"] == environment.AUTO_TOKENIZER_CLASS
    assert receipt["multimodal_auto_class"] == environment.MULTIMODAL_AUTO_CLASS
    assert receipt["packages"] == environment.PACKAGE_PINS
    assert receipt["module_origins"] == environment.MODULE_ORIGINS


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("auto_tokenizer_class", "transformers.AutoTokenizer"),
        ("multimodal_auto_class", "transformers.AutoModelForMultimodalLM"),
        ("python", {**environment.PYTHON, "entrypoint": "/usr/bin/python3"}),
        (
            "module_origins",
            {**environment.MODULE_ORIGINS, "torch": "/tmp/torch/__init__.py"},
        ),
        ("packages", {**environment.PACKAGE_PINS, "extra": "1.0"}),
    ],
)
def test_rejects_environment_identity_drift(
    tmp_path: Path, field: str, mutation: Any
) -> None:
    path, _, source = _fixture(tmp_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt[field] = mutation
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(environment.PCF1EnvironmentError, match="content differs"):
        environment.validate_environment_receipt(path, _sha256(path), source)


def test_rejects_runtime_source_tamper_after_receipt(tmp_path: Path) -> None:
    path, digest, source = _fixture(tmp_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    (Path(receipt["runtime_root"]) / source).write_text(
        "# mutated after capture\n", encoding="utf-8"
    )

    with pytest.raises(environment.PCF1EnvironmentError, match="source differs"):
        environment.validate_environment_receipt(path, digest, source)
