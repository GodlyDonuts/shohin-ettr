from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_dense_public_campaign_manifest import CampaignManifestError, run


def test_manifest_is_ordered_and_refuses_wrong_cardinality(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text("{}\n{}\n")
    second.write_text("{}\n")
    output = tmp_path / "manifest.json"
    payload = run(
        [f"mmlu_pro={first}=2=2048", f"ifeval={second}=1=1024"], output
    )
    assert payload["rows"] == 3
    assert [row["name"] for row in payload["benchmarks"]] == ["mmlu_pro", "ifeval"]
    assert json.loads(output.read_text())["status"] == "frozen"
    with pytest.raises(CampaignManifestError, match="row count differs"):
        run([f"wrong={first}=3=8"], tmp_path / "bad.json")
