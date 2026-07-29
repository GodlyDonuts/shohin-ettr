import hashlib
import json
from pathlib import Path

import pytest
import zstandard as zstd

from pipeline.materialize_domain_balanced_residual import (
    DomainBalanceError,
    materialize_domain_balanced_residual,
)
from pipeline.test_verify_tokenized_shards import build_corpus
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    DOCUMENT_LEDGER_SCHEMA,
    canonical_payload_sha256,
    sha256_file,
)
from pipeline.verify_tokenized_shards import verify_manifest


SELECTION_CODE = Path(__file__).with_name(
    "materialize_domain_balanced_residual.py"
)


def _make_source(tmp_path: Path) -> tuple[Path, Path]:
    source, source_selection = build_corpus(tmp_path, schema="v3")
    raw = b"".join(bytes((token, 0)) for token in range(1, 9))
    shard = source / "shard_00000.u16.zst"
    shard.write_bytes(zstd.ZstdCompressor(level=3).compress(raw))
    specs = (
        ("a" * 64, "1" * 64, "alpha.example", 0, 2),
        ("b" * 64, "2" * 64, "alpha.example", 2, 4),
        ("c" * 64, "3" * 64, "beta.example", 4, 6),
        ("d" * 64, "4" * 64, None, 6, 8),
    )
    rows = []
    for index, (identity, document, domain, start, end) in enumerate(specs):
        payload = raw[start * 2 : end * 2]
        rows.append(
            {
                "allowed_value": "CCBY",
                "chars": 100,
                "document_sha256": document,
                "domain": domain,
                "schema": DOCUMENT_LEDGER_SCHEMA,
                "shard": shard.name,
                "source_row_index": index,
                "stable_identity_sha256": identity,
                "token_end": end,
                "token_sha256": hashlib.sha256(payload).hexdigest(),
                "token_start": start,
                "tokens": end - start,
            }
        )
    ledger_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("ascii")
    ledger = source / DOCUMENT_LEDGER_NAME
    ledger.write_bytes(zstd.ZstdCompressor(level=3).compress(ledger_payload))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "tokens": 8,
            "kept": 4,
            "shard_files": [
                {
                    "path": shard.name,
                    "bytes": shard.stat().st_size,
                    "tokens": 8,
                    "sha256": sha256_file(shard),
                }
            ],
            "document_ledger": {
                "path": DOCUMENT_LEDGER_NAME,
                "bytes": ledger.stat().st_size,
                "sha256": sha256_file(ledger),
                "rows": 4,
                "tokens": 8,
                "contains_document_text": False,
                "schema": DOCUMENT_LEDGER_SCHEMA,
            },
        }
    )
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return source, source_selection


def _make_policy(
    tmp_path: Path,
    source: Path,
    source_selection: Path,
) -> Path:
    evidence = tmp_path / "review.receipt.json"
    evidence.write_text('{"contains_document_text":false}\n')
    manifest = json.loads((source / "manifest.json").read_text())
    policy = {
        "schema": "shohin-domain-balance-policy-v1",
        "source_manifest_payload_sha256": manifest["payload_sha256"],
        "source_selection_code_sha256": sha256_file(source_selection),
        "default_domain_token_cap": 3,
        "domain_token_cap_overrides": {},
        "reject_missing_domain": True,
        "selection_priority": "stable_identity_sha256_ascending",
        "evidence": [
            {
                "path": str(evidence),
                "bytes": evidence.stat().st_size,
                "sha256": sha256_file(evidence),
            }
        ],
    }
    policy["payload_sha256"] = canonical_payload_sha256(policy)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    return path


def test_domain_balancing_is_deterministic_and_reverifies(tmp_path: Path) -> None:
    source, source_selection = _make_source(tmp_path)
    policy = _make_policy(tmp_path, source, source_selection)
    output = tmp_path / "balanced"
    result = materialize_domain_balanced_residual(
        source_dir=source,
        source_selection_code=source_selection,
        policy_path=policy,
        selection_code=SELECTION_CODE,
        output_dir=output,
        shard_tokens=3,
    )
    assert result["documents"] == 2
    assert result["tokens"] == 4
    assert result["dropped_documents"] == 2
    receipt = verify_manifest(
        output,
        selection_code=SELECTION_CODE,
        require_external_inputs=True,
    )
    assert receipt["document_rows"] == 2
    manifest = json.loads((output / "manifest.json").read_text())
    records = {
        row["domain"]: row
        for row in manifest["domain_balanced_residual"]["domain_records"]
    }
    assert records["alpha.example"]["retained_documents"] == 1
    assert records["beta.example"]["retained_documents"] == 1
    assert records["<missing>"]["retained_documents"] == 0


def test_policy_mutation_fails_closed(tmp_path: Path) -> None:
    source, source_selection = _make_source(tmp_path)
    policy_path = _make_policy(tmp_path, source, source_selection)
    policy = json.loads(policy_path.read_text())
    policy["default_domain_token_cap"] = 4
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(DomainBalanceError, match="policy contract"):
        materialize_domain_balanced_residual(
            source_dir=source,
            source_selection_code=source_selection,
            policy_path=policy_path,
            selection_code=SELECTION_CODE,
            output_dir=tmp_path / "balanced",
        )


def test_evidence_mutation_fails_closed(tmp_path: Path) -> None:
    source, source_selection = _make_source(tmp_path)
    policy_path = _make_policy(tmp_path, source, source_selection)
    evidence = tmp_path / "review.receipt.json"
    evidence.write_text("substituted")
    with pytest.raises(DomainBalanceError, match="evidence"):
        materialize_domain_balanced_residual(
            source_dir=source,
            source_selection_code=source_selection,
            policy_path=policy_path,
            selection_code=SELECTION_CODE,
            output_dir=tmp_path / "balanced",
        )
