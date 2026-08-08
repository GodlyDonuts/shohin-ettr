from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from audit_phase2_admission_bundle import (
    ADMISSION_SCHEMA,
    Phase2AdmissionError,
    REQUIRED_CHECKS,
    REQUIRED_UTILITY,
    audit,
)
from pipeline.tokenize_shards import canonical_payload_sha256, sha256_file
from test_data_contract import _contract
from data_contract import (
    TrainingDataContractError,
    checkpoint_admission_binding,
    resolve_phase2_admission_bundle,
    resolve_training_data_contract,
)


def write_admission(
    tmp_path: Path,
    contract_path: Path,
    *,
    level: str = "canary",
    fresh: bool = True,
    utility: bool = False,
) -> Path:
    contract = json.loads(contract_path.read_text())
    corpus = contract["corpora"][0]
    manifest = json.loads((Path(corpus["path"]) / "manifest.json").read_text())
    receipts = []
    for index in range(6):
        evidence = tmp_path / f"evidence-{index}.json"
        evidence.write_text(json.dumps({"index": index}) + "\n")
        receipts.append(
            {
                "label": f"evidence-{index}",
                "path": str(evidence.resolve()),
                "sha256": sha256_file(evidence),
            }
        )
    report = {
        "schema": ADMISSION_SCHEMA,
        "status": "admitted",
        "admission_level": level,
        "corpus_name": corpus["name"],
        "manifest_payload_sha256": corpus["manifest_payload_sha256"],
        "unique_tokens": manifest["tokens"],
        "documents": manifest["kept"],
        "fresh_source": fresh,
        "checks": {name: True for name in sorted(REQUIRED_CHECKS)},
        "utility": {name: utility for name in sorted(REQUIRED_UTILITY)},
        "evidence_receipts": receipts,
    }
    report["payload_sha256"] = canonical_payload_sha256(report)
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def arguments(
    tmp_path: Path,
    *,
    level: str = "canary",
    fresh: bool = True,
    utility: bool = False,
) -> argparse.Namespace:
    contract, contract_sha256 = _contract(tmp_path)
    admission = write_admission(
        tmp_path, contract, level=level, fresh=fresh, utility=utility
    )
    manifest = json.loads(
        (Path(json.loads(contract.read_text())["corpora"][0]["path"]) / "manifest.json").read_text()
    )
    return argparse.Namespace(
        contract=contract,
        contract_sha256=contract_sha256,
        admission=[("candidate", admission)],
        level=level,
        minimum_unique_tokens=manifest["tokens"],
        deep_verify=True,
        output=tmp_path / "bundle.json",
    )


def test_canary_admission_does_not_require_completed_utility(tmp_path: Path):
    args = arguments(tmp_path)
    report = audit(args)
    assert report["training_eligible"] is True
    assert report["fresh_sampling_weight"] == 1.0
    data_resolution = resolve_training_data_contract(
        args.contract,
        expected_sha256=args.contract_sha256,
        deep_verify=True,
    )
    resolution = resolve_phase2_admission_bundle(
        args.output,
        expected_sha256=sha256_file(args.output),
        data_resolution=data_resolution,
        required_level="canary",
    )
    binding = checkpoint_admission_binding(resolution)
    assert binding["admission_level"] == "canary"
    assert binding["fresh_sampling_weight"] == 1.0


def test_non_deep_verified_bundle_is_rejected_by_trainer(tmp_path: Path):
    args = arguments(tmp_path)
    args.deep_verify = False
    audit(args)
    data_resolution = resolve_training_data_contract(
        args.contract,
        expected_sha256=args.contract_sha256,
        deep_verify=True,
    )
    with pytest.raises(
        TrainingDataContractError, match="admission payload differs"
    ):
        resolve_phase2_admission_bundle(
            args.output,
            expected_sha256=sha256_file(args.output),
            data_resolution=data_resolution,
            required_level="canary",
        )


def test_post_audit_evidence_mutation_is_rejected_by_trainer(tmp_path: Path):
    args = arguments(tmp_path)
    report = audit(args)
    data_resolution = resolve_training_data_contract(
        args.contract,
        expected_sha256=args.contract_sha256,
        deep_verify=True,
    )
    evidence_path = Path(report["corpora"]["candidate"]["evidence"][0]["path"])
    evidence_path.write_text("mutated after audit\n")
    with pytest.raises(
        TrainingDataContractError, match="admission evidence binding differs"
    ):
        resolve_phase2_admission_bundle(
            args.output,
            expected_sha256=sha256_file(args.output),
            data_resolution=data_resolution,
            required_level="canary",
        )


def test_production_admission_requires_completed_utility(tmp_path: Path):
    args = arguments(tmp_path, level="production", utility=False)
    with pytest.raises(Phase2AdmissionError, match="utility"):
        audit(args)


def test_historical_only_contract_fails_mostly_fresh_gate(tmp_path: Path):
    report = audit(arguments(tmp_path, fresh=False))
    assert report["training_eligible"] is False
    assert report["gates"]["mostly_fresh_sampling_weight"] is False


def test_evidence_substitution_fails_closed(tmp_path: Path):
    args = arguments(tmp_path)
    admission = args.admission[0][1]
    report = json.loads(admission.read_text())
    Path(report["evidence_receipts"][0]["path"]).write_text("changed\n")
    with pytest.raises(Phase2AdmissionError, match="evidence SHA-256"):
        audit(args)
