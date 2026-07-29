#!/usr/bin/env python3
"""Profile domain concentration in a verified v3 tokenized corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any, Iterable

from pipeline.build_general_source_review_packet import iter_document_ledger
from pipeline.materialize_domain_balanced_residual import (
    MISSING_DOMAIN,
    _canonical_domain,
)
from pipeline.tokenize_shards import canonical_payload_sha256, sha256_file
from pipeline.verify_tokenized_shards import verify_manifest


PROFILE_SCHEMA = "shohin-tokenized-domain-profile-v1"
DEFAULT_CAP_PROJECTIONS = (
    1_000_000_000,
    500_000_000,
    250_000_000,
    100_000_000,
    50_000_000,
    25_000_000,
    10_000_000,
    5_000_000,
)


class DomainProfileError(ValueError):
    """A verified domain profile cannot be produced."""


def profile_domains(
    *,
    source_dir: Path,
    source_selection_code: Path,
    output_path: Path,
    cap_projections: Iterable[int] = DEFAULT_CAP_PROJECTIONS,
) -> dict[str, Any]:
    if (
        not source_selection_code.is_file()
        or source_selection_code.is_symlink()
        or output_path.exists()
        or output_path.is_symlink()
    ):
        raise DomainProfileError("domain profile arguments differ")
    verification = verify_manifest(
        source_dir,
        selection_code=source_selection_code,
        require_external_inputs=True,
    )
    try:
        manifest = json.loads((source_dir / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DomainProfileError("source manifest is unreadable") from exc
    if (
        manifest.get("schema") != "shohin-tokenized-shards-v3"
        or not verification.get("document_ledger_verified")
    ):
        raise DomainProfileError("source is not a verified v3 corpus")

    domain_tokens: Counter[str] = Counter()
    domain_documents: Counter[str] = Counter()
    total_tokens = total_documents = 0
    for row in iter_document_ledger(source_dir / "documents.jsonl.zst"):
        domain = _canonical_domain(row.get("domain"))
        tokens = row.get("tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
            raise DomainProfileError("document token count differs")
        domain_tokens[domain] += tokens
        domain_documents[domain] += 1
        total_tokens += tokens
        total_documents += 1
    if (
        total_tokens != manifest.get("tokens")
        or total_documents != manifest.get("kept")
        or set(domain_tokens) != set(domain_documents)
    ):
        raise DomainProfileError("domain profile accounting differs")

    projections: list[dict[str, Any]] = []
    seen_caps: set[int] = set()
    for cap in cap_projections:
        if (
            not isinstance(cap, int)
            or isinstance(cap, bool)
            or cap < 1
            or cap in seen_caps
        ):
            raise DomainProfileError("domain cap projection differs")
        seen_caps.add(cap)
        retained = sum(
            min(tokens, cap)
            for domain, tokens in domain_tokens.items()
            if domain != MISSING_DOMAIN
        )
        projections.append(
            {
                "domain_token_cap": cap,
                "projected_tokens_after_cap_and_missing_domain_rejection": retained,
                "projected_retention_fraction": retained / total_tokens,
                "domains_over_cap": sum(
                    domain != MISSING_DOMAIN and tokens > cap
                    for domain, tokens in domain_tokens.items()
                ),
            }
        )

    ranked = domain_tokens.most_common()
    profile = {
        "schema": PROFILE_SCHEMA,
        "contains_document_text": False,
        "source_path": str(source_dir.resolve()),
        "source_manifest_payload_sha256": manifest["payload_sha256"],
        "source_selection_code_sha256": sha256_file(source_selection_code),
        "document_ledger_sha256": manifest["document_ledger"]["sha256"],
        "verification": verification,
        "documents": total_documents,
        "tokens": total_tokens,
        "domains": len(domain_tokens),
        "missing_domain_documents": domain_documents[MISSING_DOMAIN],
        "missing_domain_tokens": domain_tokens[MISSING_DOMAIN],
        "concentration": {
            f"top_{count}_token_fraction": (
                sum(tokens for _domain, tokens in ranked[:count]) / total_tokens
            )
            for count in (1, 5, 10, 25, 50, 100, 500)
        },
        "top_domains": [
            {
                "domain": domain,
                "documents": domain_documents[domain],
                "tokens": tokens,
                "token_fraction": tokens / total_tokens,
            }
            for domain, tokens in ranked[:100]
        ],
        "cap_projections": projections,
    }
    profile["payload_sha256"] = canonical_payload_sha256(profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise DomainProfileError("refusing existing partial profile")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as output:
        output.write(json.dumps(profile, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, output_path)
    return profile


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-selection-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    profile = profile_domains(
        source_dir=arguments.source_dir,
        source_selection_code=arguments.source_selection_code,
        output_path=arguments.output,
    )
    print(
        json.dumps(
            {
                "domains": profile["domains"],
                "documents": profile["documents"],
                "payload_sha256": profile["payload_sha256"],
                "tokens": profile["tokens"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
