from __future__ import annotations

import hashlib
from importlib import util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ettr_runtime_bundle import (
    ETTRRuntimeBundleError,
    ETTRRuntimeBundleReceipt,
    materialize_runtime_bundle,
)
from run_ettr_verified_stage import _VerifiedSourceFinder, _verify


TRAIN = Path(__file__).resolve().parent
BOOTSTRAP = TRAIN / "run_ettr_verified_stage.py"
MANIFEST_SCHEMA = "ettr-factorial-execution-manifest-v3"


def _write_canonical(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    path.chmod(0o444)


def _manifest(
    path: Path,
    *,
    receipt: ETTRRuntimeBundleReceipt,
) -> str:
    source_hashes = dict(receipt.source_files)
    value = {
        "bootstrap_sha256": hashlib.sha256(BOOTSTRAP.read_bytes()).hexdigest(),
        "compiler_runner_sha256": source_hashes[
            "run_ettr_world_compiler.py"
        ],
        "runtime_bundle_sha256": receipt.sha256(),
        "schema": MANIFEST_SCHEMA,
    }
    _write_canonical(path, value)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap(
    bootstrap: Path,
    *,
    manifest: Path,
    manifest_sha256: str,
    receipt: Path,
    bundle: Path,
    runner_arguments: tuple[str, ...] = ("--help",),
    isolated_flags: tuple[str, ...] = ("-I", "-S", "-B"),
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            *isolated_flags,
            str(bootstrap),
            "--manifest",
            str(manifest),
            "--manifest-sha256",
            manifest_sha256,
            "--runtime-receipt",
            str(receipt),
            "--bundle-root",
            str(bundle),
            "--stage",
            "world",
            "--",
            *runner_arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_runtime_bundle_round_trip_is_exact_and_immutable(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    receipt = materialize_runtime_bundle(TRAIN, bundle)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.canonical_bytes())
    receipt_path.chmod(0o444)

    loaded = ETTRRuntimeBundleReceipt.from_path(receipt_path)
    assert loaded == receipt
    assert loaded.sha256() == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    loaded.validate(bundle)


def test_runtime_bundle_rejects_dependency_substitution(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    receipt = materialize_runtime_bundle(TRAIN, bundle)
    target = bundle / "model.py"
    bundle.chmod(0o755)
    target.chmod(0o644)
    target.write_text("raise RuntimeError('substituted')\n", encoding="ascii")
    target.chmod(0o444)
    bundle.chmod(0o555)

    with pytest.raises(ETTRRuntimeBundleError):
        receipt.validate(bundle)


def test_copied_bootstrap_ignores_adjacent_shadow_modules(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    receipt = materialize_runtime_bundle(TRAIN, bundle)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.canonical_bytes())
    receipt_path.chmod(0o444)
    manifest_path = tmp_path / "manifest.json"
    manifest_sha256 = _manifest(manifest_path, receipt=receipt)

    attacker = tmp_path / "attacker"
    attacker.mkdir()
    copied_bootstrap = attacker / BOOTSTRAP.name
    shutil.copyfile(BOOTSTRAP, copied_bootstrap)
    marker = tmp_path / "shadow-imported"
    (attacker / "model.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('bad')\n",
        encoding="ascii",
    )
    result = _bootstrap(
        copied_bootstrap,
        manifest=manifest_path,
        manifest_sha256=manifest_sha256,
        receipt=receipt_path,
        bundle=bundle,
    )

    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_bootstrap_rejects_mutated_bundle_before_import(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    receipt = materialize_runtime_bundle(TRAIN, bundle)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.canonical_bytes())
    receipt_path.chmod(0o444)
    manifest_path = tmp_path / "manifest.json"
    manifest_sha256 = _manifest(manifest_path, receipt=receipt)
    marker = tmp_path / "mutated-imported"
    target = bundle / "model.py"
    bundle.chmod(0o755)
    target.chmod(0o644)
    target.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('bad')\n",
        encoding="ascii",
    )
    target.chmod(0o444)
    bundle.chmod(0o555)

    result = _bootstrap(
        BOOTSTRAP,
        manifest=manifest_path,
        manifest_sha256=manifest_sha256,
        receipt=receipt_path,
        bundle=bundle,
    )

    assert result.returncode != 0
    assert "runtime source differs: model.py" in result.stderr
    assert not marker.exists()


def test_bootstrap_rejects_manifest_override_and_missing_isolation(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    receipt = materialize_runtime_bundle(TRAIN, bundle)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.canonical_bytes())
    receipt_path.chmod(0o444)
    manifest_path = tmp_path / "manifest.json"
    manifest_sha256 = _manifest(manifest_path, receipt=receipt)
    other_manifest = tmp_path / "other.json"
    _write_canonical(other_manifest, {"schema": "attacker"})

    override = _bootstrap(
        BOOTSTRAP,
        manifest=manifest_path,
        manifest_sha256=manifest_sha256,
        receipt=receipt_path,
        bundle=bundle,
        runner_arguments=(
            "--execution-manifest",
            str(other_manifest),
            "--help",
        ),
    )
    missing_no_site = _bootstrap(
        BOOTSTRAP,
        manifest=manifest_path,
        manifest_sha256=manifest_sha256,
        receipt=receipt_path,
        bundle=bundle,
        isolated_flags=("-I", "-B"),
    )

    assert override.returncode != 0
    assert "runner manifest override is forbidden" in override.stderr
    assert missing_no_site.returncode != 0
    assert "requires python -I -S -B" in missing_no_site.stderr


def test_bundle_rejects_pyc_extra_hardlink_and_symlink_root(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    receipt = materialize_runtime_bundle(TRAIN, bundle)

    bundle.chmod(0o755)
    pycache = bundle / "__pycache__"
    pycache.mkdir()
    (pycache / "model.cpython-311.pyc").write_bytes(b"attacker")
    bundle.chmod(0o555)
    with pytest.raises(ETTRRuntimeBundleError):
        receipt.validate(bundle)
    bundle.chmod(0o755)
    shutil.rmtree(pycache)
    hardlink = tmp_path / "model-hardlink.py"
    hardlink.hardlink_to(bundle / "model.py")
    bundle.chmod(0o555)
    with pytest.raises(ETTRRuntimeBundleError):
        receipt.validate(bundle)
    hardlink.unlink()

    symlink_root = tmp_path / "bundle-link"
    symlink_root.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(ETTRRuntimeBundleError):
        receipt.validate(symlink_root)


def test_isolated_bootstrap_ignores_pythonpath_sitecustomize(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    receipt = materialize_runtime_bundle(TRAIN, bundle)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.canonical_bytes())
    receipt_path.chmod(0o444)
    manifest_path = tmp_path / "manifest.json"
    manifest_sha256 = _manifest(manifest_path, receipt=receipt)
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    marker = tmp_path / "sitecustomize-imported"
    (attacker / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('bad')\n",
        encoding="ascii",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(attacker)

    result = _bootstrap(
        BOOTSTRAP,
        manifest=manifest_path,
        manifest_sha256=manifest_sha256,
        receipt=receipt_path,
        bundle=bundle,
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_verified_source_loader_uses_retained_bytes_after_path_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frozen_probe.py"
    trusted = b"VALUE = 'trusted'\n"
    path.write_bytes(trusted)
    finder = _VerifiedSourceFinder(
        {"frozen_probe": (str(path), trusted)}
    )
    path.write_text("VALUE = 'attacker'\n", encoding="ascii")
    spec = finder.find_spec("frozen_probe")
    assert spec is not None
    module = util.module_from_spec(spec)
    assert spec.loader is finder
    finder.exec_module(module)
    assert module.VALUE == "trusted"


def test_verified_runner_payload_survives_post_verification_path_mutation(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    receipt = materialize_runtime_bundle(TRAIN, bundle)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.canonical_bytes())
    receipt_path.chmod(0o444)
    manifest_path = tmp_path / "manifest.json"
    manifest_sha256 = _manifest(manifest_path, receipt=receipt)

    runner, runner_bytes, _, _ = _verify(
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha256,
        receipt_path=receipt_path,
        bundle_root=bundle,
        stage="world",
    )
    expected_sha256 = dict(receipt.source_files)[runner.name]
    bundle.chmod(0o755)
    runner.chmod(0o644)
    runner.write_text("raise RuntimeError('attacker runner')\n", encoding="ascii")

    assert hashlib.sha256(runner_bytes).hexdigest() == expected_sha256
    assert b"attacker runner" not in runner_bytes
