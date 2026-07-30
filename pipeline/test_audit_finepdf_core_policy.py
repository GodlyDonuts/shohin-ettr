import hashlib
import json
from pathlib import Path

import pytest

from pipeline.audit_finepdf_core_policy import (
    FinePdfPolicyAuditError,
    audit_packet,
    canonical_payload_sha256,
)


def row(identity: str, text: str, scores, *, tokens=1000, domain="example.org"):
    document_sha = hashlib.sha256(text.encode()).hexdigest()
    return {
        "schema": "shohin-private-selected-source-review-v1",
        "dataset": "HuggingFaceFW/finepdfs-edu",
        "config": "eng_Latn",
        "stable_identity_sha256": identity,
        "document_sha256": document_sha,
        "metadata": {"fw_edu_scores": list(scores)},
        "selection": {"tokens": tokens, "domain": domain},
        "review_text": text,
    }


def write_packet(path: Path, rows):
    with path.open("w", encoding="ascii") as output:
        for value in rows:
            output.write(json.dumps(value, sort_keys=True) + "\n")


def test_audit_is_text_free_hash_bound_and_deterministic(tmp_path):
    packet = tmp_path / "private.jsonl"
    policy = Path(__file__).with_name("finepdf_core_policy.py")
    rows = [
        row("1" * 64, "A coherent educational discussion. " * 500, (2.8, 2.7)),
        row(
            "2" * 64,
            "Weekly newsletter issue 2. Parent calendar. " * 500,
            (3.0,),
        ),
        row("3" * 64, "A specialized explanation. " * 500, (1.8,)),
    ]
    write_packet(packet, rows)
    first = audit_packet(packet, review_rows_per_tier=1, policy_path=policy)
    second = audit_packet(packet, review_rows_per_tier=1, policy_path=policy)
    assert first == second
    assert first["status"] == "analysis_only_not_training_admission"
    assert first["contains_document_text"] is False
    assert {tier: value["documents"] for tier, value in first["tiers"].items()} == {
        "core": 1,
        "reject": 1,
        "residual": 1,
    }
    assert first["payload_sha256"] == canonical_payload_sha256(first)
    assert "review_text" not in json.dumps(first)


def test_duplicate_identity_fails_closed(tmp_path):
    packet = tmp_path / "private.jsonl"
    policy = Path(__file__).with_name("finepdf_core_policy.py")
    duplicate = row("1" * 64, "A document. " * 500, (2.8,))
    write_packet(packet, [duplicate, duplicate])
    with pytest.raises(FinePdfPolicyAuditError, match="identity"):
        audit_packet(packet, review_rows_per_tier=1, policy_path=policy)
