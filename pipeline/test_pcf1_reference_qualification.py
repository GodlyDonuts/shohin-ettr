"""Tests for the PCF3 trusted-reference admission boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import qualify_pcf3_references as module


def _identity_for(split: str) -> str:
    for value in range(100_000):
        identity = hashlib.sha256(f"pcf3-{split}-{value}".encode()).hexdigest()
        if module.assigned_split(identity, module.SPLIT_SEED) == split:
            return identity
    raise AssertionError(f"unable to synthesize {split}")


def test_reference_canary_skips_holdout_and_emits_no_content(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {
            "schema": "fixture",
            "identity_sha256": _identity_for(split),
            "task": "mbpp",
            "code": f"value = {index}",
            "test_setup_code": "",
            "test_list": [f"assert value == {index}"],
        }
        for index, split in enumerate(("train", "development", "holdout"))
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(module, "SOURCE_SHA256", module.sha256_file(source))
    monkeypatch.setattr(module, "qualify_allocation", lambda: {"status": "pass"})
    observed: list[str] = []

    def setups(nonsealed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        observed.extend(str(row["identity_sha256"]) for row in nonsealed)
        return [
            {
                "setup_source_sha256": hashlib.sha256(b"").hexdigest(),
                "receipt_sha256": "a" * 64,
            }
        ]

    monkeypatch.setattr(module, "qualify_mbpp_assessor_setups", setups)
    monkeypatch.setattr(
        module,
        "mbpp_allocation_setup_receipts_sha256",
        lambda _receipts: "b" * 64,
    )

    def preflight(
        row: dict[str, Any], *, split: str, setup_qualification: dict[str, Any]
    ) -> dict[str, Any]:
        assert split != "holdout"
        assert setup_qualification["receipt_sha256"] == "a" * 64
        return {
            "identity_sha256": row["identity_sha256"],
            "split": split,
            "reference_assessment_mode": "trusted_reference",
        }

    monkeypatch.setattr(module, "preflight_mbpp_reference", preflight)
    output = tmp_path / "output"
    report = module.qualify(source, output)
    assert set(observed) == {rows[0]["identity_sha256"], rows[1]["identity_sha256"]}
    assert rows[2]["identity_sha256"] not in observed
    assert report["nonsealed_reference_rows"] == 2
    assert report["split_counts"] == {"development": 1, "holdout": 1, "train": 1}
    assert report["holdout_reference_content_accesses"] == 0
    assert report["generated_candidate_policy_applied"] is False
    rendered = (output / "report.json").read_text(encoding="utf-8")
    assert "value =" not in rendered
    assert "test_list" not in rendered


def test_reference_job_uses_only_the_exact_historical_hash_exception() -> None:
    source = (Path(__file__).parent / "jobs/pcf3_reference_canary.sbatch").read_text(
        encoding="utf-8"
    )
    assert "pcf1_require_safe_input" not in source
    assert 'pcf1_require_file "$SOURCE_BANK"' in source
    assert "0b6d068b4d71f407cb234579b9278dc640df09139ea906dd0f52a6ab71e05398" in source
    assert 'pcf1_sha256 "$SOURCE_BANK"' in source
