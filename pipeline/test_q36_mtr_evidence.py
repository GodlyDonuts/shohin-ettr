from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from q36_mtr_evidence import (
    Q36MTREvidenceVerificationError,
    sha256_file,
    verify_evidence_snapshot,
)


def _snapshot(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "snapshot"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    member = artifacts / "result.json"
    member.write_text('{"result":"PASS"}\n', encoding="utf-8")
    digest = sha256_file(member)
    row = {"name": "result", "sha256": digest, "bytes": member.stat().st_size}
    tree_digest = hashlib.sha256(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    payload = {
        "artifact_sha256s": {"result": digest},
        "artifact_count": 1,
        "artifact_tree_sha256": tree_digest,
        "records": [
            {
                **row,
                "primary": str((tmp_path / "primary.json").resolve()),
                "mirror": str(member.resolve()),
            }
        ],
        "primary_mirror_hashes_exact": True,
        "write_once_snapshot": True,
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    member.chmod(0o444)
    artifacts.chmod(0o555)
    root.chmod(0o555)
    return manifest, payload


def test_q36_durable_evidence_replays_exact_tree(tmp_path: Path) -> None:
    manifest, payload = _snapshot(tmp_path)
    receipt = verify_evidence_snapshot(manifest, payload)
    assert receipt["artifact_tree_sha256"] == payload["artifact_tree_sha256"]
    assert receipt["all_hashes_verified"] is True


@pytest.mark.parametrize("mutation", ("bytes", "extra", "writable"))
def test_q36_durable_evidence_rejects_post_mirror_mutation(
    tmp_path: Path, mutation: str
) -> None:
    manifest, payload = _snapshot(tmp_path)
    root = manifest.parent
    artifacts = root / "artifacts"
    root.chmod(0o755)
    artifacts.chmod(0o755)
    member = artifacts / "result.json"
    if mutation == "bytes":
        member.chmod(0o644)
        member.write_text("tampered\n", encoding="utf-8")
        member.chmod(0o444)
    elif mutation == "extra":
        extra = artifacts / "extra.json"
        extra.write_text("{}\n", encoding="utf-8")
        extra.chmod(0o444)
    else:
        member.chmod(0o644)
    artifacts.chmod(0o555)
    root.chmod(0o555)
    with pytest.raises(Q36MTREvidenceVerificationError):
        verify_evidence_snapshot(manifest, payload)
