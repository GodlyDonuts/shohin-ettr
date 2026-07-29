import json
from pathlib import Path

import pytest

from data_contract import (
    CONTRACT_SCHEMA,
    TrainingDataContractError,
    checkpoint_binding,
    resolve_training_data_contract,
)
from build_training_data_contract import build_training_data_contract
from pipeline.materialize_v3_holdout_split import materialize_holdout_split
from pipeline.test_materialize_v3_holdout_split import (
    SEED,
    SELECTION_CODE,
    _build_source,
)
from pipeline.tokenize_shards import canonical_payload_sha256, sha256_file


def _contract(
    tmp_path: Path,
) -> tuple[Path, str]:
    source, source_selection, _rows = _build_source(tmp_path)
    split = tmp_path / "split"
    materialize_holdout_split(
        source_dir=source,
        source_selection_code=source_selection,
        selection_code=SELECTION_CODE,
        output_dir=split,
        seed=SEED,
        document_validation_bps=2_500,
        domain_validation_bps=2_500,
        shard_tokens=3,
    )
    manifest = json.loads((split / "train" / "manifest.json").read_text())
    value = {
        "schema": CONTRACT_SCHEMA,
        "corpora": [
            {
                "manifest_payload_sha256": manifest["payload_sha256"],
                "name": "candidate",
                "path": str((split / "train").resolve()),
                "role": "train",
                "selection_code_path": str(SELECTION_CODE.resolve()),
                "selection_code_sha256": sha256_file(SELECTION_CODE),
                "weight": 3.0,
            }
        ],
        "purpose": "unit_test",
    }
    value["payload_sha256"] = canonical_payload_sha256(value)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path, sha256_file(path)


def test_training_contract_resolves_only_verified_train_splits(tmp_path):
    path, digest = _contract(tmp_path)
    resolution = resolve_training_data_contract(
        path,
        expected_sha256=digest,
        deep_verify=True,
    )
    assert resolution["deep_verified"]
    assert resolution["domain_weights"] == [1.0]
    assert resolution["corpora"][0]["verification"][
        "document_ledger_verified"
    ]
    binding = checkpoint_binding(resolution)
    assert binding["contract_sha256"] == digest


def test_validation_split_is_rejected(tmp_path):
    path, _digest = _contract(tmp_path)
    value = json.loads(path.read_text())
    value["corpora"][0]["path"] = str(
        Path(value["corpora"][0]["path"]).parent / "document_validation"
    )
    manifest = json.loads(
        (Path(value["corpora"][0]["path"]) / "manifest.json").read_text()
    )
    value["corpora"][0]["manifest_payload_sha256"] = manifest[
        "payload_sha256"
    ]
    value.pop("payload_sha256")
    value["payload_sha256"] = canonical_payload_sha256(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        TrainingDataContractError,
        match="bound training split",
    ):
        resolve_training_data_contract(
            path,
            expected_sha256=sha256_file(path),
            deep_verify=False,
        )


def test_contract_hash_substitution_fails(tmp_path):
    path, _digest = _contract(tmp_path)
    with pytest.raises(
        TrainingDataContractError,
        match="SHA-256 differs",
    ):
        resolve_training_data_contract(
            path,
            expected_sha256="0" * 64,
            deep_verify=False,
        )


def test_builder_deep_verifies_before_no_replace_publication(tmp_path):
    path, _digest = _contract(tmp_path)
    source = json.loads(path.read_text())
    output = tmp_path / "built.json"
    result = build_training_data_contract(
        corpora=source["corpora"],
        purpose="matched_token_ablation",
        output=output,
    )
    assert output.is_file()
    assert result["resolution"]["deep_verified"]
    assert sha256_file(output) == result["contract_sha256"]
    with pytest.raises(FileExistsError):
        build_training_data_contract(
            corpora=source["corpora"],
            purpose="matched_token_ablation",
            output=output,
        )
