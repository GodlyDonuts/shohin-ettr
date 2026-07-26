"""Fail-closed integration audit for the R12 ETTR IL v2 specifications.

This module is CPU-only and standard-library-only. It does not generate data,
load a model, touch a checkpoint, or authorize fitting. It proves that the four
normative component documents agree on the candidate schema, split
commitments, rectangle geometry, schedule unit, transaction horizon, and
source-deletion claim boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


PROTOCOL = "R12-ETTR-IL-v2"
REPORT_SCHEMA = "r12-ettr-il-v2-spec-integration-audit-v1"

SEMANTIC_SPEC = "R12_ETTR_IL_V2_SEMANTIC_GENERATOR_SPEC.md"
MATERIALIZATION_SPEC = "R12_ETTR_IL_V2_MATERIALIZATION_SPEC.md"
ARMS_SPEC = "R12_ETTR_IL_V2_ARMS_AND_STATISTICS_SPEC.md"
CUSTODY_SPEC = "R12_ETTR_IL_V2_CUSTODY_SPEC.md"
COMPONENT_SPECS = (
    SEMANTIC_SPEC,
    MATERIALIZATION_SPEC,
    ARMS_SPEC,
    CUSTODY_SPEC,
)

SEMANTIC_MASTER_PREIMAGE = (
    b"R12_ETTR_ISOLATED_LEARNABILITY_V2|2026-07-26|semantic-generator"
)
SEMANTIC_MASTER_SHA256 = (
    "f6edaccd75ba80763540b990fcd0d1c85016e2d62a79cc3bbe328a206db925dd"
)
TOKENIZER_SHA256 = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)

EXPECTED_CANDIDATE_FIELDS = (
    "schema",
    "fold",
    "split",
    "ontology",
    "stratum",
    "theory_instance",
    "theory_pool_index",
    "worlds",
    "commands",
    "depth",
    "renderer",
    "presentations",
    "queries",
    "opaque_seed",
    "generator_ordinal",
)
EXPECTED_PRESENTATIONS = (
    "base",
    "alpha_reorder",
    "alias_split",
    "relation_reification",
    "type_twin",
    "execution_semantics_twin",
)
EXPECTED_FOLD_HASHES = (
    "a4293ae0cf972abfdfb155ad4268dceeafcb7d9ebc4df975002d08896ea65ab8",
    "6e0b45dbdb28af3684db1649767a50bae6dd5ba9d8c7fdfbae6b4edad16af425",
    "ff127ca341c8215fe4d08883d2aef9a29a3ca810adc8cf44fbe7a565c4961f68",
)
EXPECTED_FOLD_COMMITMENTS = (
    "cd21d2501e57a275267080ceec35089f5d89e8c83c4d7e3a2ac22c2a39f6eb60",
    "c8509e61b93cbac341c42a2cd73e5d58cd02edbb0eff0b06173df729d83c7d01",
    "8487125d8354be89ff15dceca987a06af2e2dfd457890b387e696002771768b5",
)


class SpecAuditError(ValueError):
    """A normative v2 component disagrees with the integrated contract."""


@dataclass(frozen=True, slots=True)
class AuditResult:
    checks: tuple[str, ...]
    component_sha256: dict[str, str]
    split_spec_sha256: str
    fold_spec_sha256: tuple[str, ...]
    fold_commitments: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": list(self.checks),
            "component_sha256": dict(sorted(self.component_sha256.items())),
            "fold_commitments": list(self.fold_commitments),
            "fold_spec_sha256": list(self.fold_spec_sha256),
            "protocol": PROTOCOL,
            "schema": REPORT_SCHEMA,
            "split_spec_sha256": self.split_spec_sha256,
            "status": "pass",
        }


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_ascii(path: Path) -> str:
    payload = path.read_bytes()
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SpecAuditError(f"{path.name} is not strict ASCII") from exc
    if "\r" in text:
        raise SpecAuditError(f"{path.name} contains CR bytes")
    return text


def _extract_preimage(text: str, label: str) -> bytes:
    pattern = re.compile(
        rf"`{re.escape(label)}-PREIMAGE-BEGIN`\n\n"
        rf"```json\n(?P<body>[^\n]*\n)```\n\n"
        rf"`{re.escape(label)}-PREIMAGE-END`"
    )
    match = pattern.search(text)
    if match is None:
        raise SpecAuditError(f"{label} preimage block is absent or noncanonical")
    payload = match.group("body").encode("ascii")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SpecAuditError(f"{label} preimage is not JSON") from exc
    if payload != canonical_json_bytes(parsed):
        raise SpecAuditError(f"{label} preimage is not canonical JSON")
    return payload


def _extract_semantic_candidate_schema(text: str) -> dict[str, Any]:
    marker = "The exact candidate tuple schema is:"
    offset = text.find(marker)
    if offset < 0:
        raise SpecAuditError("semantic candidate schema marker is absent")
    match = re.search(r"```json\n(?P<body>.*?)\n```", text[offset:], re.DOTALL)
    if match is None:
        raise SpecAuditError("semantic candidate schema block is absent")
    try:
        value = json.loads(match.group("body"))
    except json.JSONDecodeError as exc:
        raise SpecAuditError("semantic candidate schema is not JSON") from exc
    if not isinstance(value, dict):
        raise SpecAuditError("semantic candidate schema is not an object")
    return value


def _require(text: str, phrase: str, label: str) -> None:
    if phrase not in text:
        raise SpecAuditError(f"{label} is absent")


def _forbid(text: str, phrase: str, label: str) -> None:
    if phrase in text:
        raise SpecAuditError(f"{label} reappeared")


def _fold_commitment(split_hash: str, fold_hash: str) -> str:
    payload = (
        b"R12-ETTR-IL-v2"
        + b"\x00"
        + b"fold-commitment"
        + b"\x00"
        + bytes.fromhex(split_hash)
        + bytes.fromhex(fold_hash)
    )
    return sha256_bytes(payload)


def audit_specs(root: Path) -> AuditResult:
    root = root.resolve()
    texts = {
        name: _read_ascii(root / name)
        for name in COMPONENT_SPECS
    }
    semantic = texts[SEMANTIC_SPEC]
    materialization = texts[MATERIALIZATION_SPEC]
    arms = texts[ARMS_SPEC]
    custody = texts[CUSTODY_SPEC]
    checks: list[str] = ["strict_ascii_components"]

    if sha256_bytes(SEMANTIC_MASTER_PREIMAGE) != SEMANTIC_MASTER_SHA256:
        raise AssertionError("internal semantic master constant differs")
    _require(
        semantic,
        SEMANTIC_MASTER_PREIMAGE.decode("ascii"),
        "semantic master preimage",
    )
    _require(semantic, SEMANTIC_MASTER_SHA256, "semantic master digest")
    checks.append("semantic_master_commitment")

    for name, text in texts.items():
        _require(text, TOKENIZER_SHA256, f"{name} tokenizer digest")
    checks.append("tokenizer_identity_shared")

    split_payload = _extract_preimage(custody, "SPLIT-SPEC")
    split_value = json.loads(split_payload)
    split_hash = sha256_bytes(split_payload)
    if tuple(split_value["candidate_tuple_fields"]) != EXPECTED_CANDIDATE_FIELDS:
        raise SpecAuditError("custody candidate fields differ")
    if split_value["candidate_tuple_schema"] != "r12-ettr-il-v2-candidate":
        raise SpecAuditError("custody candidate schema identity differs")
    if tuple(split_value["presentations"]) != EXPECTED_PRESENTATIONS:
        raise SpecAuditError("custody presentation population differs")
    _require(custody, f"Byte count: `{len(split_payload)}`", "split byte count")
    _require(custody, f"`{split_hash}`", "split digest")
    checks.append("literal_split_preimage")

    fold_payloads = tuple(
        _extract_preimage(custody, f"FOLD-{fold}")
        for fold in range(3)
    )
    fold_hashes = tuple(sha256_bytes(payload) for payload in fold_payloads)
    if fold_hashes != EXPECTED_FOLD_HASHES:
        raise SpecAuditError("fold preimage digests differ")
    commitments = tuple(
        _fold_commitment(split_hash, fold_hash)
        for fold_hash in fold_hashes
    )
    if commitments != EXPECTED_FOLD_COMMITMENTS:
        raise SpecAuditError("derived fold commitments differ")
    for commitment in commitments:
        _require(custody, commitment, "displayed fold commitment")
    checks.append("literal_fold_preimages_and_commitments")

    candidate_schema = _extract_semantic_candidate_schema(semantic)
    required = tuple(candidate_schema.get("required", ()))
    if required != tuple(sorted(EXPECTED_CANDIDATE_FIELDS)):
        raise SpecAuditError(
            "semantic candidate required fields are not the sorted custody fields"
        )
    properties = candidate_schema.get("properties")
    if not isinstance(properties, dict):
        raise SpecAuditError("semantic candidate properties are absent")
    schema_const = properties.get("schema", {}).get("const")
    if schema_const != split_value["candidate_tuple_schema"]:
        raise SpecAuditError("semantic/custody candidate schema identity differs")
    presentation_enum = (
        properties.get("presentations", {})
        .get("items", {})
        .get("enum")
    )
    if tuple(presentation_enum or ()) != EXPECTED_PRESENTATIONS:
        raise SpecAuditError("semantic/custody presentation enum differs")
    checks.append("semantic_custody_candidate_schema")

    _require(
        semantic,
        'WORLD source for semantic world `Ww` uses\n'
        '`cell_salt="world-<c>"`',
        "semantic cell-local WORLD source",
    )
    _require(
        semantic,
        'COMMAND source for semantic command `Cc` uses\n'
        '`cell_salt="command-<w>"`',
        "semantic cell-local COMMAND source",
    )
    _require(
        materialization,
        'WORLD semantics `Ww` with opaque-name `cell_salt="world-<c>"`',
        "materialization cell-local WORLD source",
    )
    _require(
        materialization,
        'COMMAND semantics `Cc` with `cell_salt="command-<w>"`',
        "materialization cell-local COMMAND source",
    )
    _require(
        materialization,
        "independently parse and canonicalize both variants",
        "cell-local semantic identity audit",
    )
    checks.append("cell_local_factor_surface_variants")

    _require(
        semantic,
        "schedule unit is one admitted invariant pair",
        "semantic pair schedule",
    )
    _require(
        arms,
        "| invariant pairs | `1` | `4` |",
        "arms pair schedule",
    )
    _require(
        materialization,
        "each 32-row microbatch has exactly 16 alignment",
        "materialization pair alignment",
    )
    _forbid(
        semantic,
        "A global update has eight causal rectangles = 32 rows.",
        "retired causal-rectangle schedule",
    )
    checks.append("paired_semantic_training_schedule")

    _require(
        materialization,
        "transaction_targets = ETTRTransactionTargets[B=16M,K=64]",
        "materialization transaction horizon",
    )
    _require(
        arms,
        "Transaction targets have exactly 64 positions",
        "arms transaction horizon",
    )
    _forbid(
        arms,
        "Transaction targets have exactly eight positions",
        "retired eight-position horizon",
    )
    checks.append("transaction_horizon_64")

    if 96 * 3 != 288 or 288 * 4 != 1152:
        raise AssertionError("internal fitting geometry differs")
    if 1152 * 2 != 2304 or 2304 * 16 != 36864:
        raise AssertionError("internal fold geometry differs")
    if 6000 * 4 != 24000 or 24000 * 2 != 48000:
        raise AssertionError("internal exposure geometry differs")
    _require(semantic, "Per fit ontology this is 288 cores", "core geometry")
    _require(arms, "Each fold has exactly `576` fit semantic cores", "fold cores")
    _require(arms, "`24,000` invariant", "pair exposures")
    checks.append("geometry_and_exposure_arithmetic")

    _forbid(
        split_payload.decode("ascii"),
        "ambiguity_deleted_twin",
        "causal-population ambiguity presentation",
    )
    _require(
        semantic,
        "`ambiguity_deleted_twin` receives no row",
        "ambiguity exclusion",
    )
    checks.append("answer_only_causal_population")

    _require(
        materialization,
        "interface-level non-consumption during fitting",
        "fitting deletion boundary",
    )
    _require(
        custody,
        "physically separated under the signed supervisor",
        "autonomous physical deletion boundary",
    )
    checks.append("source_deletion_claim_boundary")

    component_hashes = {
        name: sha256_bytes((root / name).read_bytes())
        for name in COMPONENT_SPECS
    }
    return AuditResult(
        checks=tuple(checks),
        component_sha256=component_hashes,
        split_spec_sha256=split_hash,
        fold_spec_sha256=fold_hashes,
        fold_commitments=commitments,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = audit_specs(args.root)
    except (OSError, SpecAuditError, KeyError, TypeError, ValueError) as exc:
        print(f"{REPORT_SCHEMA}: fail: {exc}")
        return 1
    payload = canonical_json_bytes(result.as_dict())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(payload.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
