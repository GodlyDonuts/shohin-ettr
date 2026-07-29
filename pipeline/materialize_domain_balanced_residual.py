#!/usr/bin/env python3
"""Materialize a deterministic domain-balanced residual from a verified v3 corpus."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from pipeline.build_general_source_review_packet import iter_document_ledger
from pipeline.materialize_cross_source_exact_residual import (
    _source_shard_bytes,
    _write_shard,
)
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    DocumentLedgerWriter,
    canonical_payload_sha256,
    sha256_file,
)
from pipeline.verify_tokenized_shards import verify_manifest


POLICY_SCHEMA = "shohin-domain-balance-policy-v1"
RESIDUAL_SCHEMA = "shohin-domain-balanced-residual-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MISSING_DOMAIN = "<missing>"


class DomainBalanceError(ValueError):
    """The domain policy cannot produce a verified residual."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DomainBalanceError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise DomainBalanceError(f"{label} is not an object")
    return value


def _verify_file_receipt(record: Mapping[str, Any], label: str) -> None:
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise DomainBalanceError(f"{label} path is missing")
    path = Path(path_value)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DomainBalanceError(f"{label} is unavailable") from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or metadata.st_nlink != 1
        or metadata.st_size != record.get("bytes")
        or sha256_file(path) != record.get("sha256")
    ):
        raise DomainBalanceError(f"{label} receipt differs")


def _verified_policy(
    policy_path: Path,
    *,
    source_manifest: Mapping[str, Any],
    source_selection_code: Path,
) -> dict[str, Any]:
    policy = _load_json(policy_path, "domain policy")
    claimed = policy.get("payload_sha256")
    unsigned = dict(policy)
    unsigned.pop("payload_sha256", None)
    default_cap = policy.get("default_domain_token_cap")
    overrides = policy.get("domain_token_cap_overrides")
    evidence = policy.get("evidence")
    if (
        policy.get("schema") != POLICY_SCHEMA
        or not isinstance(claimed, str)
        or HEX64.fullmatch(claimed) is None
        or canonical_payload_sha256(unsigned) != claimed
        or policy.get("source_manifest_payload_sha256")
        != source_manifest.get("payload_sha256")
        or policy.get("source_selection_code_sha256")
        != source_manifest.get("selection_code_sha256")
        or sha256_file(source_selection_code)
        != source_manifest.get("selection_code_sha256")
        or not isinstance(default_cap, int)
        or isinstance(default_cap, bool)
        or default_cap < 1
        or policy.get("reject_missing_domain") is not True
        or policy.get("selection_priority")
        != "stable_identity_sha256_ascending"
        or not isinstance(overrides, dict)
        or not isinstance(evidence, list)
        or not evidence
    ):
        raise DomainBalanceError("domain policy contract differs")
    for domain, cap in overrides.items():
        if (
            not isinstance(domain, str)
            or not domain
            or domain == MISSING_DOMAIN
            or not isinstance(cap, int)
            or isinstance(cap, bool)
            or cap < 0
        ):
            raise DomainBalanceError("domain cap override differs")
    for index, record in enumerate(evidence):
        if not isinstance(record, dict):
            raise DomainBalanceError("policy evidence record differs")
        _verify_file_receipt(record, f"policy evidence {index}")
    return policy


def _canonical_domain(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return MISSING_DOMAIN
    return value.strip().lower()


def _select_domain_balanced_identities(
    source_dir: Path,
    policy: Mapping[str, Any],
) -> tuple[set[str], list[dict[str, Any]], dict[str, int]]:
    entries: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in iter_document_ledger(source_dir / DOCUMENT_LEDGER_NAME):
        identity = row.get("stable_identity_sha256")
        tokens = row.get("tokens")
        if (
            not isinstance(identity, str)
            or HEX64.fullmatch(identity) is None
            or not isinstance(tokens, int)
            or isinstance(tokens, bool)
            or tokens < 1
        ):
            raise DomainBalanceError("source ledger identity differs")
        entries[_canonical_domain(row.get("domain"))].append((identity, tokens))

    default_cap = int(policy["default_domain_token_cap"])
    overrides = {
        str(domain): int(cap)
        for domain, cap in policy["domain_token_cap_overrides"].items()
    }
    selected: set[str] = set()
    records: list[dict[str, Any]] = []
    totals = {
        "input_documents": 0,
        "input_tokens": 0,
        "retained_documents": 0,
        "retained_tokens": 0,
        "dropped_documents": 0,
        "dropped_tokens": 0,
    }
    for domain in sorted(entries):
        rows = sorted(entries[domain], key=lambda item: item[0])
        cap = 0 if domain == MISSING_DOMAIN else overrides.get(domain, default_cap)
        retained_documents = retained_tokens = 0
        for identity, tokens in rows:
            if retained_tokens + tokens <= cap:
                selected.add(identity)
                retained_documents += 1
                retained_tokens += tokens
        input_tokens = sum(tokens for _identity, tokens in rows)
        record = {
            "domain": domain,
            "token_cap": cap,
            "input_documents": len(rows),
            "input_tokens": input_tokens,
            "retained_documents": retained_documents,
            "retained_tokens": retained_tokens,
            "dropped_documents": len(rows) - retained_documents,
            "dropped_tokens": input_tokens - retained_tokens,
        }
        records.append(record)
        for key in totals:
            totals[key] += int(record[key])
    if (
        not selected
        or totals["retained_documents"] != len(selected)
        or totals["input_documents"]
        != totals["retained_documents"] + totals["dropped_documents"]
        or totals["input_tokens"]
        != totals["retained_tokens"] + totals["dropped_tokens"]
    ):
        raise DomainBalanceError("domain selection accounting differs")
    return selected, records, totals


def materialize_domain_balanced_residual(
    *,
    source_dir: Path,
    source_selection_code: Path,
    policy_path: Path,
    selection_code: Path,
    output_dir: Path,
    shard_tokens: int = 100_000_000,
) -> dict[str, Any]:
    if (
        shard_tokens < 1
        or not source_selection_code.is_file()
        or source_selection_code.is_symlink()
        or not selection_code.is_file()
        or selection_code.is_symlink()
        or not policy_path.is_file()
        or policy_path.is_symlink()
    ):
        raise DomainBalanceError("domain residual arguments differ")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing existing output: {output_dir}")
    source_verification = verify_manifest(
        source_dir,
        selection_code=source_selection_code,
        require_external_inputs=True,
    )
    source_manifest = _load_json(source_dir / "manifest.json", "source manifest")
    if (
        source_manifest.get("schema") != "shohin-tokenized-shards-v3"
        or not source_verification.get("document_ledger_verified")
    ):
        raise DomainBalanceError("source is not a verified v3 corpus")
    policy = _verified_policy(
        policy_path,
        source_manifest=source_manifest,
        source_selection_code=source_selection_code,
    )
    selected, domain_records, totals = _select_domain_balanced_identities(
        source_dir,
        policy,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.partial-",
            dir=output_dir.parent,
        )
    )
    try:
        ledger = DocumentLedgerWriter(staging / DOCUMENT_LEDGER_NAME)
        shard_records: list[dict[str, Any]] = []
        output_payload = bytearray()
        output_shard = 0
        source_shard: str | None = None
        source_payload = b""
        retained_rows = retained_tokens = 0
        seen_selected: set[str] = set()
        for row in iter_document_ledger(source_dir / DOCUMENT_LEDGER_NAME):
            identity = str(row["stable_identity_sha256"])
            if identity not in selected:
                continue
            if row["shard"] != source_shard:
                source_shard = str(row["shard"])
                source_payload = _source_shard_bytes(source_dir, source_shard)
            start = int(row["token_start"]) * 2
            end = int(row["token_end"]) * 2
            document_payload = source_payload[start:end]
            if (
                len(document_payload) != int(row["tokens"]) * 2
                or hashlib.sha256(document_payload).hexdigest()
                != row["token_sha256"]
            ):
                raise DomainBalanceError("retained document token span differs")
            token_start = len(output_payload) // 2
            output_payload.extend(document_payload)
            output_row = dict(row)
            output_row.update(
                {
                    "domain": _canonical_domain(row.get("domain")),
                    "shard": f"shard_{output_shard:05d}.u16.zst",
                    "token_start": token_start,
                    "token_end": token_start + row["tokens"],
                }
            )
            ledger.write(output_row)
            seen_selected.add(identity)
            retained_rows += 1
            retained_tokens += int(row["tokens"])
            if len(output_payload) // 2 >= shard_tokens:
                shard_records.append(
                    _write_shard(
                        staging,
                        index=output_shard,
                        payload=output_payload,
                    )
                )
                output_payload = bytearray()
                output_shard += 1
        if output_payload:
            shard_records.append(
                _write_shard(
                    staging,
                    index=output_shard,
                    payload=output_payload,
                )
            )
        ledger_receipt = ledger.close()
        if (
            seen_selected != selected
            or retained_rows != totals["retained_documents"]
            or retained_tokens != totals["retained_tokens"]
            or ledger_receipt["rows"] != retained_rows
            or ledger_receipt["tokens"] != retained_tokens
            or sum(item["tokens"] for item in shard_records) != retained_tokens
        ):
            raise DomainBalanceError("residual accounting differs")

        manifest = {
            key: value
            for key, value in source_manifest.items()
            if key
            not in {
                "payload_sha256",
                "selection_code_sha256",
                "tokens",
                "shards",
                "shard_files",
                "document_ledger",
                "kept",
            }
        }
        manifest.update(
            {
                "schema": "shohin-tokenized-shards-v3",
                "selection_code_sha256": sha256_file(selection_code),
                "tokens": retained_tokens,
                "shards": len(shard_records),
                "shard_files": shard_records,
                "document_ledger": ledger_receipt,
                "kept": retained_rows,
                "dropped_domain_balance": totals["dropped_documents"],
                "dropped_domain_balance_tokens": totals["dropped_tokens"],
                "domain_balanced_residual": {
                    "schema": RESIDUAL_SCHEMA,
                    "source_path": str(source_dir.resolve()),
                    "source_manifest_payload_sha256": source_manifest[
                        "payload_sha256"
                    ],
                    "source_selection_code_sha256": source_manifest[
                        "selection_code_sha256"
                    ],
                    "source_verification": source_verification,
                    "policy_path": str(policy_path.resolve()),
                    "policy_sha256": sha256_file(policy_path),
                    "policy_payload_sha256": policy["payload_sha256"],
                    "selection_priority": policy["selection_priority"],
                    "default_domain_token_cap": policy[
                        "default_domain_token_cap"
                    ],
                    "reject_missing_domain": True,
                    "domain_records": domain_records,
                    "totals": totals,
                },
            }
        )
        manifest["payload_sha256"] = canonical_payload_sha256(manifest)
        manifest_path = staging / "manifest.json"
        with manifest_path.open("x") as destination:
            json.dump(manifest, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        verification = verify_manifest(
            staging,
            selection_code=selection_code,
            require_external_inputs=True,
        )
        if (
            verification["tokens"] != retained_tokens
            or verification["document_rows"] != retained_rows
        ):
            raise DomainBalanceError("published residual verification differs")
        os.replace(staging, output_dir)
        return {
            "schema": RESIDUAL_SCHEMA,
            "output": str(output_dir),
            "manifest_payload_sha256": manifest["payload_sha256"],
            "documents": retained_rows,
            "tokens": retained_tokens,
            "dropped_documents": totals["dropped_documents"],
            "dropped_tokens": totals["dropped_tokens"],
            "domains": len(domain_records),
            "verification": verification,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-selection-code", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--selection-code", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    arguments = parser.parse_args(argv)
    result = materialize_domain_balanced_residual(
        source_dir=arguments.source_dir,
        source_selection_code=arguments.source_selection_code,
        policy_path=arguments.policy,
        selection_code=arguments.selection_code,
        output_dir=arguments.output_dir,
        shard_tokens=arguments.shard_tokens,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
