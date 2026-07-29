"""Verify and resolve an immutable Phase 2 general-training data contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.tokenize_shards import (  # noqa: E402
    canonical_payload_sha256,
    file_receipt,
)
from pipeline.verify_tokenized_shards import verify_manifest  # noqa: E402


CONTRACT_SCHEMA = "shohin-general-training-data-contract-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CORPUS_FIELDS = {
    "manifest_payload_sha256",
    "name",
    "path",
    "role",
    "selection_code_path",
    "selection_code_sha256",
    "weight",
}


class TrainingDataContractError(ValueError):
    """A training data contract is not exact, immutable, and training-only."""


def _load_contract(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if HEX64.fullmatch(expected_sha256) is None:
        raise TrainingDataContractError("data-contract SHA-256 is malformed")
    receipt = file_receipt(path)
    if receipt["sha256"] != expected_sha256:
        raise TrainingDataContractError("data-contract SHA-256 differs")
    try:
        contract = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingDataContractError("data contract is unreadable") from exc
    if not isinstance(contract, dict):
        raise TrainingDataContractError("data contract is not an object")
    claimed = contract.get("payload_sha256")
    unsigned = dict(contract)
    unsigned.pop("payload_sha256", None)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or not isinstance(claimed, str)
        or HEX64.fullmatch(claimed) is None
        or canonical_payload_sha256(unsigned) != claimed
    ):
        raise TrainingDataContractError("data-contract payload differs")
    return contract, receipt


def resolve_training_data_contract(
    path: Path,
    *,
    expected_sha256: str,
    deep_verify: bool,
) -> dict[str, Any]:
    contract, receipt = _load_contract(
        path,
        expected_sha256=expected_sha256,
    )
    corpora = contract.get("corpora")
    if not isinstance(corpora, list) or not corpora:
        raise TrainingDataContractError("data contract has no corpora")
    names: set[str] = set()
    paths: set[Path] = set()
    tokenizer_identity: tuple[object, object] | None = None
    resolved: list[dict[str, Any]] = []
    for index, value in enumerate(corpora):
        if not isinstance(value, dict) or set(value) != CORPUS_FIELDS:
            raise TrainingDataContractError(
                f"corpus {index} fields differ"
            )
        name = value["name"]
        role = value["role"]
        weight = value["weight"]
        manifest_hash = value["manifest_payload_sha256"]
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or role != "train"
            or not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0
            or not isinstance(manifest_hash, str)
            or HEX64.fullmatch(manifest_hash) is None
        ):
            raise TrainingDataContractError(
                f"corpus {index} identity/role/weight differs"
            )
        corpus_path = Path(str(value["path"]))
        selection_path = Path(str(value["selection_code_path"]))
        if (
            not corpus_path.is_absolute()
            or not corpus_path.is_dir()
            or corpus_path.is_symlink()
            or corpus_path in paths
            or ".partial" in corpus_path.name
        ):
            raise TrainingDataContractError(
                f"corpus {index} path is not an immutable final directory"
            )
        selection_receipt = file_receipt(selection_path)
        if (
            selection_receipt["sha256"]
            != value["selection_code_sha256"]
        ):
            raise TrainingDataContractError(
                f"corpus {index} selection code differs"
            )
        try:
            manifest = json.loads(
                (corpus_path / "manifest.json").read_text()
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingDataContractError(
                f"corpus {index} manifest is unreadable"
            ) from exc
        holdout = manifest.get("holdout_split")
        tokenizer = manifest.get("tokenizer")
        current_tokenizer = (
            tokenizer.get("sha256") if isinstance(tokenizer, dict) else None,
            tokenizer.get("vocab_size")
            if isinstance(tokenizer, dict)
            else None,
        )
        if (
            manifest.get("schema") != "shohin-tokenized-shards-v3"
            or manifest.get("payload_sha256") != manifest_hash
            or manifest.get("selection_code_sha256")
            != selection_receipt["sha256"]
            or not isinstance(holdout, dict)
            or holdout.get("name") != "train"
            or not isinstance(tokenizer, dict)
            or current_tokenizer[0] is None
            or current_tokenizer[1] is None
        ):
            raise TrainingDataContractError(
                f"corpus {index} is not the bound training split"
            )
        if tokenizer_identity is None:
            tokenizer_identity = current_tokenizer
        elif tokenizer_identity != current_tokenizer:
            raise TrainingDataContractError(
                "training corpora use different tokenizer identities"
            )
        verification = (
            verify_manifest(
                corpus_path,
                selection_code=selection_path,
                require_external_inputs=True,
            )
            if deep_verify
            else None
        )
        names.add(name)
        paths.add(corpus_path)
        resolved.append(
            {
                "name": name,
                "path": str(corpus_path),
                "weight": float(weight),
                "manifest_payload_sha256": manifest_hash,
                "selection_code_sha256": selection_receipt["sha256"],
                "verification": verification,
            }
        )
    weight_total = sum(item["weight"] for item in resolved)
    normalized_weights = [
        item["weight"] / weight_total for item in resolved
    ]
    return {
        "schema": "shohin-general-training-data-contract-resolution-v1",
        "contract": receipt,
        "contract_payload_sha256": contract["payload_sha256"],
        "deep_verified": deep_verify,
        "corpora": resolved,
        "shard_dirs": [item["path"] for item in resolved],
        "domain_weights": normalized_weights,
        "tokenizer_sha256": tokenizer_identity[0],
        "tokenizer_vocab_size": tokenizer_identity[1],
    }


def checkpoint_binding(resolution: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract_sha256": resolution["contract"]["sha256"],
        "contract_payload_sha256": resolution["contract_payload_sha256"],
        "corpus_manifest_payload_sha256s": [
            item["manifest_payload_sha256"]
            for item in resolution["corpora"]
        ],
        "normalized_domain_weights": list(resolution["domain_weights"]),
        "tokenizer_sha256": resolution["tokenizer_sha256"],
        "tokenizer_vocab_size": resolution["tokenizer_vocab_size"],
    }
