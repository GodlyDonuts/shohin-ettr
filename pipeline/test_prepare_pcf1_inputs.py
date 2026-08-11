"""Direct tests for the sole legacy-input PCF1 preparation boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

import prepare_pcf1_inputs as prepare_module
from prepare_pcf1_inputs import (
    MODEL_REVISION,
    PCF1PrepareError,
    copy_exact,
    prepare,
    validate_model_snapshot,
    verify_model_manifest,
)


def test_exact_hash_mismatch_fails_without_destination(tmp_path: Path) -> None:
    source = tmp_path / "legacy.bin"
    destination = tmp_path / "safe.bin"
    source.write_bytes(b"wrong")
    with pytest.raises(PCF1PrepareError, match="source hash differs"):
        copy_exact(source, destination, "0" * 64)
    assert not destination.exists()


@pytest.mark.parametrize("term", ("holdout", "product", "public"))
def test_prepare_output_firewall_precedes_input_reads(
    tmp_path: Path, term: str
) -> None:
    missing = tmp_path / "missing"
    args = argparse.Namespace(
        model_revision=MODEL_REVISION,
        output=tmp_path / term / "pcf1",
        assessor_output=tmp_path / "custodian" / "confirmation_assessors.jsonl",
        assessor_receipt_output=tmp_path / "custodian" / "assessor_receipt.json",
        cpu_receipt_output=tmp_path / "custodian" / "prepare_receipt.json",
        pairs=missing,
        math_bank=missing,
        science_bank=missing,
        code_bank=missing,
        b1_data=missing,
        model_root=missing,
        environment_receipt=missing,
    )
    with pytest.raises(PCF1PrepareError, match="safe output path"):
        prepare(args)


def test_model_manifest_rejects_symlink_snapshot_members(tmp_path: Path) -> None:
    blobs = tmp_path / "blobs"
    model = tmp_path / "snapshot"
    blobs.mkdir()
    model.mkdir()
    (blobs / "config").write_bytes(b'{"model_type":"fixture"}\n')
    (blobs / "weight").write_bytes(b"weight-bytes")
    (model / "config.json").symlink_to(blobs / "config")
    (model / "model.safetensors").symlink_to(blobs / "weight")
    manifest = model / "SHA256SUMS"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256((model / name).read_bytes()).hexdigest()}  {name}\n"
            for name in ("config.json", "model.safetensors")
        )
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(PCF1PrepareError, match="model tree differs"):
        verify_model_manifest(
            model,
            expected_sha256=digest,
            expected_files=2,
            expected_bytes=len((blobs / "config").read_bytes())
            + len((blobs / "weight").read_bytes()),
        )


def _model_config() -> dict[str, object]:
    return {
        "model_type": "mistral3",
        "architectures": ["Mistral3ForConditionalGeneration"],
        "text_config": {
            "model_type": "ministral3",
            "hidden_size": 4096,
            "num_hidden_layers": 34,
        },
        "vision_config": {"model_type": "pixtral"},
    }


def test_model_snapshot_rejects_wrong_config_and_path(tmp_path: Path) -> None:
    wrong_path = tmp_path / MODEL_REVISION
    wrong_path.mkdir()
    (wrong_path / "config.json").write_text(json.dumps(_model_config()))
    with pytest.raises(PCF1PrepareError, match="repository/revision path"):
        validate_model_snapshot(wrong_path, MODEL_REVISION)

    snapshot = (
        tmp_path
        / "models--mistralai--Ministral-3-8B-Reasoning-2512"
        / "snapshots"
        / MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    wrong = _model_config()
    wrong["text_config"] = {**wrong["text_config"], "num_hidden_layers": 33}
    (snapshot / "config.json").write_text(json.dumps(wrong))
    with pytest.raises(PCF1PrepareError, match="config/layout"):
        validate_model_snapshot(snapshot, MODEL_REVISION, expected_root=snapshot)


def _fixture_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> argparse.Namespace:
    inputs = tmp_path / "legacy"
    inputs.mkdir()
    paths = {}
    for name in ("pairs", "math", "science", "code"):
        path = inputs / f"{name}.jsonl"
        path.write_text(f"{name}\n")
        paths[name] = path
    b1 = inputs / "b1.jsonl"
    b1.write_bytes(b"exact-b1-fixture\n")
    monkeypatch.setattr(
        prepare_module, "B1_SHA256", hashlib.sha256(b1.read_bytes()).hexdigest()
    )
    model = (
        tmp_path
        / "models--mistralai--Ministral-3-8B-Reasoning-2512"
        / "snapshots"
        / MODEL_REVISION
    )
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps(_model_config()) + "\n")
    (model / "weights.safetensors").write_bytes(b"weights")
    (model / "SOURCE_REVISION").write_text(MODEL_REVISION + "\n")
    manifest = model / "SHA256SUMS"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256((model / name).read_bytes()).hexdigest()}  {name}\n"
            for name in ("SOURCE_REVISION", "config.json", "weights.safetensors")
        )
    )
    monkeypatch.setattr(prepare_module, "MODEL_ROOT", model)
    monkeypatch.setattr(
        prepare_module,
        "MODEL_CONFIG_SHA256",
        hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        prepare_module,
        "qualify_allocation",
        lambda: {"schema": "shohin-pcf1-code-sandbox-receipt-v1", "status": "pass"},
    )
    monkeypatch.setattr(
        prepare_module,
        "MODEL_SOURCE_REVISION_SHA256",
        hashlib.sha256((model / "SOURCE_REVISION").read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        prepare_module,
        "MODEL_MANIFEST_SHA256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(prepare_module, "MODEL_MANIFEST_FILES", 3)
    monkeypatch.setattr(
        prepare_module,
        "MODEL_MANIFEST_BYTES",
        sum(
            (model / name).stat().st_size
            for name in ("SOURCE_REVISION", "config.json", "weights.safetensors")
        ),
    )
    environment = tmp_path / "environment_receipt.json"
    environment.write_text(
        json.dumps(
            {
                "schema": "shohin-pcf1-environment-receipt-v1",
                "status": "complete",
            }
        )
        + "\n"
    )

    def fake_freeze_sources(
        *, output: Path, assessor_output: Path, assessor_receipt_output: Path, **_kwargs
    ):
        output.mkdir()
        (output / "train_sources.jsonl").write_text("{}\n")
        (output / "development_sources.jsonl").write_text("{}\n")
        (output / "reference_sandbox_receipt.json").write_text(
            json.dumps(
                {
                    "schema": "shohin-pcf1-code-sandbox-receipt-v1",
                    "status": "pass",
                }
            )
        )
        assessor_output.parent.mkdir(parents=True, exist_ok=True)
        assessor_output.write_text('{"assessor":"sealed-fixture"}\n')
        assessor_receipt = {
            "schema": "shohin-pcf1-confirmation-assessor-receipt-v1",
            "status": "complete",
            "board_sha256": hashlib.sha256(assessor_output.read_bytes()).hexdigest(),
            "rows": 1289,
            "semantic_access": "final_score_only",
        }
        assessor_receipt_output.write_text(json.dumps(assessor_receipt))
        report = {
            "schema": "shohin-pcf1-data-freeze-report-v1",
            "status": "complete",
            "outputs": {
                "confirmation_assessor_receipt": {
                    "sha256": hashlib.sha256(
                        assessor_receipt_output.read_bytes()
                    ).hexdigest(),
                    "board_sha256": hashlib.sha256(
                        assessor_output.read_bytes()
                    ).hexdigest(),
                    "rows": 1,
                }
            },
        }
        (output / "report.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(prepare_module, "freeze_sources", fake_freeze_sources)
    return argparse.Namespace(
        model_revision=MODEL_REVISION,
        output=tmp_path / "safe" / "pcf1",
        pairs=paths["pairs"],
        math_bank=paths["math"],
        science_bank=paths["science"],
        code_bank=paths["code"],
        b1_data=b1,
        model_root=model,
        environment_receipt=environment,
        assessor_output=tmp_path / "custodian" / "confirmation_assessors.jsonl",
        assessor_receipt_output=tmp_path / "custodian" / "assessor_receipt.json",
        cpu_receipt_output=tmp_path / "custodian" / "prepare_receipt.json",
    )


def test_assessor_is_separate_and_prepare_is_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture_args(tmp_path, monkeypatch)
    original_sha256_file = prepare_module.sha256_file

    def reject_postfreeze_board_read(path: Path) -> str:
        if Path(path).resolve() == args.assessor_output.resolve():
            raise AssertionError("prepare reopened the frozen assessor board")
        return original_sha256_file(path)

    monkeypatch.setattr(prepare_module, "sha256_file", reject_postfreeze_board_read)
    receipt = prepare(args)
    board = args.assessor_output
    assert board.is_file()
    assert not board.is_relative_to(args.output)
    assert args.assessor_receipt_output.is_file()
    assessor_receipt = json.loads(args.assessor_receipt_output.read_text())
    assert set(assessor_receipt) == {
        "schema",
        "status",
        "board_sha256",
        "rows",
        "semantic_access",
    }
    assert str(board.resolve()) not in json.dumps(assessor_receipt)
    assert args.cpu_receipt_output.is_file()
    assert "confirmation_assessors" not in receipt["outputs"]
    assert all("assessor" not in key.casefold() for key in receipt["outputs"])
    assert hashlib.sha256(board.read_bytes()).hexdigest() not in json.dumps(receipt)
    frozen = json.loads((args.output / "sources" / "report.json").read_text())
    assert "confirmation_assessor_receipt" in frozen["outputs"]
    assert str(board.resolve()) not in json.dumps(frozen)
    custodian = json.loads(args.cpu_receipt_output.read_text())
    assert custodian["confirmation_assessors"]["path"] == str(board.resolve())
    assert custodian["confirmation_assessors"]["gpu_exported"] is False
    with pytest.raises(PCF1PrepareError, match="refusing existing"):
        prepare(args)


def test_prepare_rejects_assessor_receipt_path_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture_args(tmp_path, monkeypatch)
    frozen_freeze = prepare_module.freeze_sources

    def leaking_freeze(**kwargs: object) -> dict[str, object]:
        report = frozen_freeze(**kwargs)
        receipt_path = Path(kwargs["assessor_receipt_output"])
        receipt = json.loads(receipt_path.read_text())
        receipt["board"] = str(Path(kwargs["assessor_output"]).resolve())
        receipt_path.write_text(json.dumps(receipt))
        report_path = Path(kwargs["output"]) / "report.json"
        report["outputs"]["confirmation_assessor_receipt"]["sha256"] = hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest()
        report_path.write_text(json.dumps(report))
        return report

    monkeypatch.setattr(prepare_module, "freeze_sources", leaking_freeze)
    with pytest.raises(PCF1PrepareError, match="assessor receipt differs"):
        prepare(args)
