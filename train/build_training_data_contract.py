#!/usr/bin/env python3
"""Build and deep-verify an immutable Phase 2 training-data contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

from data_contract import CONTRACT_SCHEMA, resolve_training_data_contract
from pipeline.tokenize_shards import (
    canonical_payload_sha256,
    file_receipt,
)


NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _parse_corpus(value: str) -> dict[str, Any]:
    try:
        left, selection, weight_text = value.split("::")
        name, path_text = left.split("=", 1)
        weight = float(weight_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected name=/absolute/train::/absolute/selection.py::weight"
        ) from exc
    path = Path(path_text)
    selection_path = Path(selection)
    if (
        NAME.fullmatch(name) is None
        or not path.is_absolute()
        or not selection_path.is_absolute()
        or weight <= 0
    ):
        raise argparse.ArgumentTypeError(
            "corpus name, paths, or positive weight differ"
        )
    try:
        manifest = json.loads((path / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(
            f"unreadable corpus manifest: {path}"
        ) from exc
    selection_receipt = file_receipt(selection_path)
    return {
        "manifest_payload_sha256": manifest.get("payload_sha256"),
        "name": name,
        "path": str(path),
        "role": "train",
        "selection_code_path": str(selection_path),
        "selection_code_sha256": selection_receipt["sha256"],
        "weight": weight,
    }


def build_training_data_contract(
    *,
    corpora: list[dict[str, Any]],
    purpose: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    if not corpora or not purpose:
        raise ValueError("training-data contract arguments differ")
    contract = {
        "schema": CONTRACT_SCHEMA,
        "purpose": purpose,
        "corpora": corpora,
    }
    contract["payload_sha256"] = canonical_payload_sha256(contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.partial-",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as destination:
            json.dump(contract, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        physical_sha256 = file_receipt(temporary)["sha256"]
        resolution = resolve_training_data_contract(
            temporary,
            expected_sha256=physical_sha256,
            deep_verify=True,
        )
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(f"refusing existing output: {output}") from None
        temporary.unlink()
        return {
            "contract": contract,
            "contract_sha256": physical_sha256,
            "resolution": resolution,
        }
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", type=_parse_corpus, required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = build_training_data_contract(
        corpora=arguments.corpus,
        purpose=arguments.purpose,
        output=arguments.output,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
