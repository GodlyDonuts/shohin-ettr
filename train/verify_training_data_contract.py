#!/usr/bin/env python3
"""Deep-verify a Phase 2 training-data contract and write a no-replace receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from data_contract import resolve_training_data_contract
from pipeline.tokenize_shards import canonical_payload_sha256


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"refusing existing output: {arguments.output}")
    resolution = resolve_training_data_contract(
        arguments.contract,
        expected_sha256=arguments.contract_sha256,
        deep_verify=True,
    )
    report = {
        "schema": "shohin-general-training-data-contract-verification-v1",
        "resolution": resolution,
        "training_eligible": True,
    }
    report["payload_sha256"] = canonical_payload_sha256(report)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
