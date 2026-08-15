import hashlib
import json
from pathlib import Path

import pytest

import read_upward_moe_data_receipt as module
from hf_upward_moe_generate_drafts import host_spec


def _fixture(tmp_path: Path, kind: str = "revision_train"):
    spec = host_spec("nemotron-super")
    data = tmp_path / f"{kind}.jsonl"
    data.write_text('{"row":1}\n', encoding="utf-8")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "shohin-q36-mtr-data-report-v1",
                "status": "complete",
                "model_revision": spec.model_revision,
                "draft_host": spec.host,
                "source_disjoint": True,
                "model_owned_drafts": True,
                "sealed_access": {"holdout": 0, "product": 0, "public": 0},
                "outputs": {
                    kind: {
                        "path": str(data.resolve()),
                        "sha256": digest,
                        "rows": 9655 if kind == "revision_train" else 1289,
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report, data, digest


def test_receipt_resolves_hash_for_dependency_prestaging(tmp_path: Path) -> None:
    report, data, digest = _fixture(tmp_path)
    assert (
        module.resolve_hash("nemotron-super", report, "revision_train", data) == digest
    )


def test_receipt_rejects_cross_host_or_tampered_data(tmp_path: Path) -> None:
    report, data, _ = _fixture(tmp_path, "development")
    with pytest.raises(module.UpwardMoEDataReceiptError):
        module.resolve_hash("mixtral-8x22b", report, "development", data)
    data.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(module.UpwardMoEDataReceiptError):
        module.resolve_hash("nemotron-super", report, "development", data)
