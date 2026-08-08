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
ADMISSION_BUNDLE_SCHEMA = "shohin-phase2-admission-bundle-v1"
ADMISSION_LEVELS = {"canary": 0, "production": 1}
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


def resolve_phase2_admission_bundle(
    path: Path,
    *,
    expected_sha256: str,
    data_resolution: Mapping[str, Any],
    required_level: str,
) -> dict[str, Any]:
    if required_level not in ADMISSION_LEVELS:
        raise TrainingDataContractError("Phase-2 admission level differs")
    if HEX64.fullmatch(expected_sha256) is None:
        raise TrainingDataContractError("Phase-2 admission SHA-256 is malformed")
    receipt = file_receipt(path)
    if receipt["sha256"] != expected_sha256:
        raise TrainingDataContractError("Phase-2 admission SHA-256 differs")
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingDataContractError("Phase-2 admission is unreadable") from exc
    if not isinstance(report, dict):
        raise TrainingDataContractError("Phase-2 admission is not an object")
    claimed = report.get("payload_sha256")
    unsigned = dict(report)
    unsigned.pop("payload_sha256", None)
    level = report.get("requested_level")
    if (
        report.get("schema") != ADMISSION_BUNDLE_SCHEMA
        or report.get("status") != "admitted"
        or report.get("training_eligible") is not True
        or report.get("deep_verified") is not True
        or not isinstance(claimed, str)
        or HEX64.fullmatch(claimed) is None
        or canonical_payload_sha256(unsigned) != claimed
        or level not in ADMISSION_LEVELS
        or ADMISSION_LEVELS[level] < ADMISSION_LEVELS[required_level]
    ):
        raise TrainingDataContractError("Phase-2 admission payload differs")
    gates = report.get("gates")
    if not isinstance(gates, dict) or not gates or not all(
        value is True for value in gates.values()
    ):
        raise TrainingDataContractError("Phase-2 admission gates differ")
    if (
        data_resolution.get("deep_verified") is not True
        or
        report.get("contract", {}).get("sha256")
        != data_resolution["contract"]["sha256"]
        or report.get("contract_payload_sha256")
        != data_resolution["contract_payload_sha256"]
        or report.get("tokenizer_sha256") != data_resolution["tokenizer_sha256"]
        or report.get("tokenizer_vocab_size")
        != data_resolution["tokenizer_vocab_size"]
        or report.get("normalized_domain_weights")
        != data_resolution["domain_weights"]
    ):
        raise TrainingDataContractError("Phase-2 admission contract/tokenizer differs")
    corpora = report.get("corpora")
    expected_corpora = {
        item["name"]: item["manifest_payload_sha256"]
        for item in data_resolution["corpora"]
    }
    if not isinstance(corpora, dict) or set(corpora) != set(expected_corpora):
        raise TrainingDataContractError("Phase-2 admission corpus coverage differs")
    for name, manifest_sha256 in expected_corpora.items():
        value = corpora[name]
        admission_path = Path(str(value.get("path", ""))) if isinstance(value, dict) else Path()
        if (
            not isinstance(value, dict)
            or value.get("manifest_payload_sha256") != manifest_sha256
            or not admission_path.is_absolute()
            or not admission_path.is_file()
            or admission_path.is_symlink()
            or not isinstance(value.get("sha256"), str)
            or HEX64.fullmatch(value["sha256"]) is None
            or file_receipt(admission_path)["sha256"] != value["sha256"]
            or not isinstance(value.get("payload_sha256"), str)
            or HEX64.fullmatch(value["payload_sha256"]) is None
            or not isinstance(value.get("unique_tokens"), int)
            or value["unique_tokens"] <= 0
        ):
            raise TrainingDataContractError("Phase-2 admission corpus binding differs")
        try:
            corpus_admission = json.loads(admission_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingDataContractError(
                "Phase-2 corpus admission is unreadable"
            ) from exc
        unsigned_admission = dict(corpus_admission)
        unsigned_admission.pop("payload_sha256", None)
        if (
            corpus_admission.get("payload_sha256") != value["payload_sha256"]
            or canonical_payload_sha256(unsigned_admission)
            != value["payload_sha256"]
        ):
            raise TrainingDataContractError(
                "Phase-2 corpus admission payload differs"
            )
        evidence = value.get("evidence")
        if not isinstance(evidence, list) or len(evidence) < 6:
            raise TrainingDataContractError(
                "Phase-2 admission evidence is incomplete"
            )
        for evidence_receipt in evidence:
            if not isinstance(evidence_receipt, dict):
                raise TrainingDataContractError(
                    "Phase-2 admission evidence binding differs"
                )
            evidence_path = Path(str(evidence_receipt.get("path", "")))
            evidence_sha256 = evidence_receipt.get("sha256")
            if (
                not evidence_path.is_absolute()
                or not evidence_path.is_file()
                or evidence_path.is_symlink()
                or not isinstance(evidence_sha256, str)
                or HEX64.fullmatch(evidence_sha256) is None
                or file_receipt(evidence_path)["sha256"] != evidence_sha256
            ):
                raise TrainingDataContractError(
                    "Phase-2 admission evidence binding differs"
                )
    if (
        not isinstance(report.get("unique_tokens"), int)
        or report["unique_tokens"] <= 0
        or not isinstance(report.get("minimum_unique_tokens"), int)
        or report["unique_tokens"] < report["minimum_unique_tokens"]
        or not isinstance(report.get("fresh_sampling_weight"), (int, float))
        or isinstance(report.get("fresh_sampling_weight"), bool)
        or report["fresh_sampling_weight"] < 0.70
    ):
        raise TrainingDataContractError("Phase-2 admission token/freshness gate differs")
    return {
        "schema": "shohin-phase2-admission-resolution-v1",
        "receipt": receipt,
        "payload_sha256": claimed,
        "level": level,
        "unique_tokens": report["unique_tokens"],
        "minimum_unique_tokens": report["minimum_unique_tokens"],
        "fresh_sampling_weight": float(report["fresh_sampling_weight"]),
        "corpus_admission_sha256s": {
            name: corpora[name]["sha256"] for name in sorted(corpora)
        },
    }


def checkpoint_admission_binding(resolution: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "admission_sha256": resolution["receipt"]["sha256"],
        "admission_payload_sha256": resolution["payload_sha256"],
        "admission_level": resolution["level"],
        "unique_tokens": resolution["unique_tokens"],
        "minimum_unique_tokens": resolution["minimum_unique_tokens"],
        "fresh_sampling_weight": resolution["fresh_sampling_weight"],
        "corpus_admission_sha256s": resolution["corpus_admission_sha256s"],
    }
