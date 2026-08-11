#!/usr/bin/env python3
"""Freeze nonsealed PCF1 sources and bind later model-owned drafts.

The ``freeze`` phase reads the immutable CVG1 pair/source universe, writes
only train and development views, and reduces the holdout to a count plus an
ordered-identity digest.  The ``materialize`` phase accepts exact nonsealed
model drafts and writes revision training, training-only commit evaluation,
and development evaluation rows.  Neither phase accepts a public, product,
or holdout model-visible/output path; ``freeze`` alone admits the exact
hash-pinned historical ``product_reasoning`` source directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Iterator, Mapping

PAIR_SCHEMA = "shohin-cvg1-whole-lineage-pairs-v1"
TRAIN_SOURCE_SCHEMA = "shohin-pcf1-train-source-v1"
DEVELOPMENT_SOURCE_SCHEMA = "shohin-pcf1-development-source-v1"
ASSESSOR_SCHEMA = "shohin-pcf1-assessor-v1"
CONFIRMATION_ASSESSOR_SCHEMA = "shohin-pcf1-confirmation-assessor-v1"
CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA = "shohin-pcf1-confirmation-assessor-receipt-v1"
SEALED_RECEIPT_SCHEMA = "shohin-pcf1-sealed-identity-receipt-v1"
FREEZE_REPORT_SCHEMA = "shohin-pcf1-data-freeze-report-v1"
DRAFT_SCHEMA = "shohin-pcf1-model-draft-v1"
REVISION_TRAIN_SCHEMA = "shohin-pcf1-revision-train-v1"
EVAL_SCHEMA = "shohin-pcf1-eval-v1"
MATERIALIZATION_REPORT_SCHEMA = "shohin-pcf1-data-report-v1"

SPLIT_SEED = 2026080811
SPLITS = ("train", "development", "holdout")
NONSEALED_SPLITS = ("train", "development")
TASKS = ("math500", "bbh_logic", "mbpp")
OUTCOMES = ("base_only", "both_correct", "both_wrong", "expert_only")
FORBIDDEN_PATH_TERMS = ("holdout", "product", "public")
LEGACY_SOURCE_SCHEMA = "shohin-product-rollout-bank-v1"

PAIR_SHA256 = "45f1d66ce5e87dc2a1f4c3594bdde2bae26e9417e879d16eb4eddb228b696afe"
SOURCE_BANK_SHA256S = frozenset(
    {
        "e0ede83257e441050a019f59fb13d9c85bd6cba1d6a755ab86fb7129966ddbe5",
        "5a96859fd9088cde598b61da60dd2c6cb7281323ee06c034742a1b4e0e237017",
        "0b6d068b4d71f407cb234579b9278dc640df09139ea906dd0f52a6ab71e05398",
    }
)
B1_DRAFT_TRAINING_SHA256 = (
    "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549"
)
EXPECTED_SPLIT_COUNTS = {"train": 5824, "development": 1289, "holdout": 1279}
EXPECTED_REVISION_PRESENTATIONS = 9655
PINNED_MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"


class PCF1DataError(RuntimeError):
    """A PCF1 source, split, draft, or custody invariant differs."""


@dataclass(frozen=True)
class CustodyContract:
    """Hash and geometry pins used by the CPU freeze.

    Tests may supply a complete synthetic contract; the command line always
    uses ``FROZEN_CUSTODY`` and therefore has no fixture/override escape hatch.
    """

    pairs_sha256: str
    bank_sha256s: frozenset[str]
    draft_training_sha256: str
    split_seed: int
    split_counts: Mapping[str, int]
    revision_presentations: int


FROZEN_CUSTODY = CustodyContract(
    pairs_sha256=PAIR_SHA256,
    bank_sha256s=SOURCE_BANK_SHA256S,
    draft_training_sha256=B1_DRAFT_TRAINING_SHA256,
    split_seed=SPLIT_SEED,
    split_counts=EXPECTED_SPLIT_COUNTS,
    revision_presentations=EXPECTED_REVISION_PRESENTATIONS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def assigned_split(identity: str, seed: int = SPLIT_SEED) -> str:
    """Apply the frozen PCJ1/IDR1 NUL-delimited 70/15/15 split."""

    if not _is_sha256(identity):
        raise PCF1DataError("PCF1 source identity is invalid")
    digest = hashlib.sha256(f"{seed}\0{identity}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 7_000:
        return "train"
    if bucket < 8_500:
        return "development"
    return "holdout"


def ordered_identity_sha256(identities: Iterable[str]) -> str:
    ordered = sorted(identities)
    return hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest()


def _validate_path(path: Path, label: str) -> None:
    rendered = str(path.expanduser().resolve(strict=False)).casefold()
    term = next((term for term in FORBIDDEN_PATH_TERMS if term in rendered), None)
    if term is not None:
        raise PCF1DataError(f"refusing {label} path containing {term}: {path}")


def _validate_frozen_input_path(path: Path, label: str) -> bool:
    """Allow only the historical ``product_reasoning`` input directory."""

    parts = tuple(part.casefold() for part in path.expanduser().resolve().parts)
    for term in ("holdout", "public"):
        if any(term in part for part in parts):
            raise PCF1DataError(f"refusing {label} path containing {term}: {path}")
    product_parts = [part for part in parts if "product" in part]
    if any(part != "product_reasoning" for part in product_parts):
        raise PCF1DataError(f"refusing {label} path containing product: {path}")
    return bool(product_parts)


def _validate_schema(value: object, expected: str | None, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PCF1DataError(f"{label} schema is invalid")
    lowered = value.casefold()
    if any(term in lowered for term in FORBIDDEN_PATH_TERMS):
        raise PCF1DataError(f"{label} schema crosses the PCF1 firewall")
    if expected is not None and value != expected:
        raise PCF1DataError(f"{label} schema differs")
    return value


def _validate_source_schema(value: object, bank_sha256: str) -> str:
    """Admit the historical source schema only behind an exact bank hash."""

    if value == LEGACY_SOURCE_SCHEMA:
        if not _is_sha256(bank_sha256):
            raise PCF1DataError("PCF1 legacy source lacks exact hash custody")
        return LEGACY_SOURCE_SCHEMA
    return _validate_schema(value, None, "PCF1 source")


def _iter_jsonl(path: Path, label: str) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise PCF1DataError(f"missing {label}: {path}")
    seen = False
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            seen = True
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PCF1DataError(
                    f"malformed {label} JSONL at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise PCF1DataError(f"non-object {label} row at line {line_number}")
            yield row
    if not seen:
        raise PCF1DataError(f"empty {label}: {path}")


def _source_prompt(source: Mapping[str, Any]) -> str:
    task = source.get("task")
    if task == "mbpp":
        text = source.get("text")
        tests = source.get("test_list")
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(tests, list)
            or not tests
            or any(not isinstance(test, str) or not test.strip() for test in tests)
        ):
            raise PCF1DataError("PCF1 executable source is incomplete")
        return (
            "Write Python code that solves the task and passes every test. Return "
            "only executable Python code, without Markdown fences.\n\nTask:\n"
            f"{text}\n\nTests:\n" + "\n".join(tests)
        )
    question = source.get("question")
    if not isinstance(question, str) or not question.strip():
        raise PCF1DataError("PCF1 natural source question is empty")
    if task == "bbh_logic":
        return (
            f"{question}\n\nReason carefully, then put only the exact requested "
            "answer or option label inside \\boxed{}."
        )
    return question


def _canonical_target(source: Mapping[str, Any]) -> str:
    target = (
        source.get("code") if source.get("task") == "mbpp" else source.get("answer")
    )
    if not isinstance(target, str) or not target.strip():
        raise PCF1DataError("PCF1 verified source target is empty")
    return target.strip()


def _assessor(source: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only fields required by the exact scorer and self-refinement prompt."""

    task = source.get("task")
    identity = source.get("identity_sha256")
    if task not in TASKS or not _is_sha256(identity):
        raise PCF1DataError("PCF1 assessor identity or task differs")
    if task == "mbpp":
        reference = source.get("reference_execution_sha256")
        if reference is not None and not _is_sha256(reference):
            raise PCF1DataError("PCF1 execution receipt is invalid")
        assessor: dict[str, Any] = {
            "schema": ASSESSOR_SCHEMA,
            "identity_sha256": identity,
            "task": task,
            "text": source["text"],
            "test_list": source["test_list"],
            "test_setup_code": str(source.get("test_setup_code") or ""),
        }
        if reference is not None:
            assessor["reference_execution_sha256"] = reference
        return assessor
    assessor = {
        "schema": ASSESSOR_SCHEMA,
        "identity_sha256": identity,
        "task": task,
        "question": source["question"],
        "answer": _canonical_target(source),
    }
    expected = source.get("expected_answer_normalized")
    if expected is not None:
        if not isinstance(expected, str) or not expected:
            raise PCF1DataError("PCF1 normalized answer is invalid")
        assessor["expected_answer_normalized"] = expected
    return assessor


def _training_target(
    pair: Mapping[str, Any], source: Mapping[str, Any]
) -> tuple[str, str]:
    candidates = pair.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise PCF1DataError("PCF1 pair requires two frozen candidates")
    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise PCF1DataError("PCF1 candidate is invalid")
        completion = candidate.get("completion")
        correct = candidate.get("correct")
        lineage = candidate.get("lineage")
        if (
            not isinstance(completion, str)
            or not completion.strip()
            or not isinstance(correct, bool)
            or not isinstance(lineage, str)
            or not lineage
        ):
            raise PCF1DataError("PCF1 candidate custody is incomplete")
        normalized.append(
            {"completion": completion.strip(), "correct": correct, "lineage": lineage}
        )
    if [candidate["lineage"] for candidate in normalized] != ["base", "expert"]:
        raise PCF1DataError("PCF1 frozen candidate lineage order differs")
    expected_outcome = {
        (True, False): "base_only",
        (True, True): "both_correct",
        (False, False): "both_wrong",
        (False, True): "expert_only",
    }[tuple(bool(candidate["correct"]) for candidate in normalized)]
    if pair.get("outcome_class") != expected_outcome:
        raise PCF1DataError("PCF1 candidate outcome binding differs")
    correct = [candidate for candidate in normalized if candidate["correct"]]
    if len(correct) == 1:
        return str(correct[0]["completion"]), "verified_candidate"
    if len(correct) == 2:
        chosen = min(correct, key=lambda row: (len(row["completion"]), row["lineage"]))
        return str(chosen["completion"]), "shortest_verified_candidate"
    return _canonical_target(source), "source_verified_repair"


def revision_presentations(outcome_class: object) -> int:
    if outcome_class not in OUTCOMES:
        raise PCF1DataError("PCF1 revision outcome class differs")
    return 4 if outcome_class in {"base_only", "expert_only"} else 1


def revision_prompt(source_prompt: str, draft: str) -> str:
    serialized_draft = draft if draft.strip() else "<EMPTY_DRAFT>"
    return (
        "Solve the original problem by checking and revising the model's earlier draft. "
        "The draft may contain useful steps or errors; do not merely critique it.\n\n"
        f"Original problem:\n{source_prompt}\n\nInternal draft:\n{serialized_draft}\n\n"
        "Follow the original problem's requested output format.\n\n"
        f"Original problem:\n{source_prompt}"
    )


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with path.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _validate_contract(contract: CustodyContract, split_seed: int) -> None:
    if split_seed != SPLIT_SEED or split_seed != contract.split_seed:
        raise PCF1DataError("PCF1 split seed differs from the frozen contract")
    if not _is_sha256(contract.pairs_sha256):
        raise PCF1DataError("PCF1 pair custody hash is invalid")
    if not contract.bank_sha256s or any(
        not _is_sha256(value) for value in contract.bank_sha256s
    ):
        raise PCF1DataError("PCF1 bank custody hashes are invalid")
    if not _is_sha256(contract.draft_training_sha256):
        raise PCF1DataError("PCF1 draft-training reference hash is invalid")
    if set(contract.split_counts) != set(SPLITS) or any(
        not isinstance(value, int) or value <= 0
        for value in contract.split_counts.values()
    ):
        raise PCF1DataError("PCF1 frozen split geometry is invalid")
    if (
        not isinstance(contract.revision_presentations, int)
        or contract.revision_presentations < contract.split_counts["train"]
        or contract.revision_presentations > 4 * contract.split_counts["train"]
    ):
        raise PCF1DataError("PCF1 revision presentation geometry is invalid")


def freeze_sources(
    *,
    pairs_path: Path,
    bank_paths: list[Path],
    output: Path,
    assessor_output: Path,
    assessor_receipt_output: Path,
    reference_evaluator: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
    reference_sandbox_receipt: dict[str, Any] | None = None,
    split_seed: int = SPLIT_SEED,
    contract: CustodyContract = FROZEN_CUSTODY,
) -> dict[str, Any]:
    """Write the nonsealed source views and a content-free sealed receipt."""

    _validate_contract(contract, split_seed)
    if contract == FROZEN_CUSTODY and (
        reference_evaluator is None or reference_sandbox_receipt is None
    ):
        raise PCF1DataError(
            "PCF1 frozen source publication requires MBPP reference preflight"
        )
    if (reference_evaluator is None) != (reference_sandbox_receipt is None):
        raise PCF1DataError("PCF1 reference preflight inputs are incomplete")
    _validate_path(output, "output")
    _validate_path(assessor_output, "confirmation-assessor output")
    _validate_path(assessor_receipt_output, "confirmation-assessor receipt output")
    if not pairs_path.is_file():
        raise PCF1DataError(f"missing PCF1 pair bank: {pairs_path}")
    if not bank_paths or any(not path.is_file() for path in bank_paths):
        raise PCF1DataError("one or more PCF1 source banks are missing")
    pairs_historical_path = _validate_frozen_input_path(pairs_path, "pair")
    bank_historical_paths = [
        _validate_frozen_input_path(path, "source-bank") for path in bank_paths
    ]
    if sha256_file(pairs_path) != contract.pairs_sha256:
        raise PCF1DataError("PCF1 pair SHA-256 differs")
    bank_hashes = [sha256_file(path) for path in bank_paths]
    if (
        len(bank_hashes) != len(set(bank_hashes))
        or frozenset(bank_hashes) != contract.bank_sha256s
    ):
        raise PCF1DataError("PCF1 source-bank SHA-256 set differs")
    historical_directory_hashes = {
        *([contract.pairs_sha256] if pairs_historical_path else []),
        *[
            digest
            for digest, historical in zip(
                bank_hashes, bank_historical_paths, strict=True
            )
            if historical
        ],
    }

    pairs: dict[str, dict[str, Any]] = {}
    split_identities: dict[str, set[str]] = {split: set() for split in SPLITS}
    split_counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for row in _iter_jsonl(pairs_path, "PCF1 pair bank"):
        _validate_schema(row.get("schema"), PAIR_SCHEMA, "PCF1 pair")
        identity = row.get("identity_sha256")
        if not _is_sha256(identity) or identity in pairs:
            raise PCF1DataError("PCF1 pair identity is invalid or duplicated")
        task = row.get("task")
        if task not in TASKS:
            raise PCF1DataError("PCF1 pair task differs")
        split = assigned_split(identity, split_seed)
        split_identities[split].add(identity)
        split_counts[split] += 1
        if split != "holdout":
            task_counts[split][str(task)] += 1
        # Holdout content is deliberately not retained past this iteration.
        pairs[identity] = (
            {"identity_sha256": identity, "task": task, "split": split}
            if split == "holdout"
            else row
        )

    if dict(split_counts) != dict(contract.split_counts):
        raise PCF1DataError("PCF1 split geometry differs")
    if any(
        split_identities[left] & split_identities[right]
        for left, right in (
            ("train", "development"),
            ("train", "holdout"),
            ("development", "holdout"),
        )
    ):
        raise PCF1DataError("PCF1 source partitions overlap")

    train_rows: list[dict[str, Any]] = []
    development_rows: list[dict[str, Any]] = []
    confirmation_assessor_rows: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    legacy_schema_bank_hashes: set[str] = set()
    reference_preflight_rows: list[dict[str, Any]] = []
    for bank_path, bank_sha256 in zip(bank_paths, bank_hashes, strict=True):
        for source in _iter_jsonl(bank_path, "PCF1 source bank"):
            source_schema = _validate_source_schema(source.get("schema"), bank_sha256)
            if source_schema == LEGACY_SOURCE_SCHEMA:
                legacy_schema_bank_hashes.add(bank_sha256)
            identity = source.get("identity_sha256")
            if not _is_sha256(identity) or identity in seen_sources:
                raise PCF1DataError("PCF1 source identity is invalid or duplicated")
            pair = pairs.get(identity)
            if pair is None:
                raise PCF1DataError("PCF1 source identity escapes the pair universe")
            seen_sources.add(identity)
            task = source.get("task")
            if task != pair.get("task"):
                raise PCF1DataError("PCF1 pair/source task binding differs")
            split = assigned_split(identity, split_seed)
            if split == "holdout":
                # Do not inspect or copy question, answer, assessor, or trajectory fields.
                continue
            if task == "mbpp" and reference_evaluator is not None:
                try:
                    reference = reference_evaluator(source, split)
                except Exception as error:
                    raise PCF1DataError(
                        "PCF1 nonsealed MBPP reference preflight failed"
                    ) from error
                expected_keys = {
                    "identity_sha256",
                    "split",
                    "candidate_source_sha256",
                    "program_sha256",
                    "setup_source_sha256",
                    "setup_qualification_sha256",
                    "candidate_policy_sha256",
                    "sandbox_config_sha256",
                    "allocation_probe_sha256",
                    "termination_classification",
                }
                if (
                    not isinstance(reference, dict)
                    or set(reference) != expected_keys
                    or reference.get("identity_sha256") != identity
                    or reference.get("split") != split
                    or any(
                        not _is_sha256(reference.get(field))
                        for field in (
                            "candidate_source_sha256",
                            "program_sha256",
                            "setup_source_sha256",
                            "setup_qualification_sha256",
                            "candidate_policy_sha256",
                            "sandbox_config_sha256",
                            "allocation_probe_sha256",
                        )
                    )
                    or reference.get("termination_classification")
                    != "trusted_tests_completed"
                ):
                    raise PCF1DataError("PCF1 nonsealed MBPP reference receipt differs")
                reference_preflight_rows.append(reference)
            prompt = _source_prompt(source)
            question = pair.get("question")
            if not isinstance(question, str) or question not in prompt:
                raise PCF1DataError("PCF1 pair/source question binding differs")
            if split == "train":
                target, target_kind = _training_target(pair, source)
                train_rows.append(
                    {
                        "schema": TRAIN_SOURCE_SCHEMA,
                        "identity_sha256": identity,
                        "split": "train",
                        "task": task,
                        "outcome_class": pair.get("outcome_class"),
                        "source_prompt": prompt,
                        "response": target,
                        "target_kind": target_kind,
                        "assessor": _assessor(source),
                        "runtime_fields": ["source_prompt"],
                        "supervisor_only_fields": [
                            "response",
                            "target_kind",
                            "assessor",
                            "task",
                            "outcome_class",
                        ],
                    }
                )
            else:
                # Validate both frozen candidates without admitting them to runtime.
                _training_target(pair, source)
                development_rows.append(
                    {
                        "schema": DEVELOPMENT_SOURCE_SCHEMA,
                        "identity_sha256": identity,
                        "split": "development",
                        "task": task,
                        "source_prompt": prompt,
                        "runtime_fields": ["source_prompt"],
                        "supervisor_only_fields": ["task"],
                    }
                )
                confirmation_assessor_rows.append(
                    {
                        "schema": CONFIRMATION_ASSESSOR_SCHEMA,
                        "identity_sha256": identity,
                        "split": "confirmation",
                        "task": task,
                        "assessor": _assessor(source),
                    }
                )
    if seen_sources != set(pairs):
        raise PCF1DataError("PCF1 pair/source identity coverage differs")

    train_rows.sort(key=lambda row: row["identity_sha256"])
    development_rows.sort(key=lambda row: row["identity_sha256"])
    confirmation_assessor_rows.sort(key=lambda row: row["identity_sha256"])
    if (
        len(train_rows) != split_counts["train"]
        or len(development_rows) != split_counts["development"]
        or len(confirmation_assessor_rows) != split_counts["development"]
    ):
        raise PCF1DataError("PCF1 nonsealed source coverage differs")

    reference_preflight_rows.sort(key=lambda row: row["identity_sha256"])
    expected_reference_rows = sum(
        task_counts[split].get("mbpp", 0) for split in NONSEALED_SPLITS
    )
    if reference_evaluator is not None:
        if (
            len(reference_preflight_rows) != expected_reference_rows
            or len({row["identity_sha256"] for row in reference_preflight_rows})
            != expected_reference_rows
            or len(
                {
                    (
                        row["candidate_policy_sha256"],
                        row["sandbox_config_sha256"],
                        row["allocation_probe_sha256"],
                    )
                    for row in reference_preflight_rows
                }
            )
            != 1
        ):
            raise PCF1DataError("PCF1 MBPP reference preflight coverage differs")
        reference_receipt_lines = b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode()
            for row in reference_preflight_rows
        )
        setup_qualifications = sorted(
            {
                (
                    row["setup_source_sha256"],
                    row["setup_qualification_sha256"],
                )
                for row in reference_preflight_rows
            }
        )
        setup_qualification_lines = b"".join(
            (
                json.dumps(
                    {
                        "setup_source_sha256": setup_sha256,
                        "setup_qualification_sha256": qualification_sha256,
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode()
            for setup_sha256, qualification_sha256 in setup_qualifications
        )
        reference_policy, reference_config, reference_probe = (
            reference_preflight_rows[0]["candidate_policy_sha256"],
            reference_preflight_rows[0]["sandbox_config_sha256"],
            reference_preflight_rows[0]["allocation_probe_sha256"],
        )
        reference_preflight = {
            "schema": "shohin-pcf1-mbpp-reference-preflight-v1",
            "status": "pass",
            "scope": ["train", "development"],
            "rows": expected_reference_rows,
            "ordered_identity_sha256": ordered_identity_sha256(
                row["identity_sha256"] for row in reference_preflight_rows
            ),
            "row_receipts_sha256": hashlib.sha256(reference_receipt_lines).hexdigest(),
            "unique_setups": len(setup_qualifications),
            "setup_pair_receipts_sha256": hashlib.sha256(
                setup_qualification_lines
            ).hexdigest(),
            "candidate_policy_sha256": reference_policy,
            "sandbox_config_sha256": reference_config,
            "allocation_probe_sha256": reference_probe,
            "all_policy_accepted": True,
            "all_sandbox_passed": True,
            "holdout_reference_content_accesses": 0,
        }
    else:
        reference_preflight = {
            "schema": "shohin-pcf1-mbpp-reference-preflight-v1",
            "status": "synthetic_contract_not_executed",
            "scope": ["train", "development"],
            "rows": expected_reference_rows,
            "unique_setups": 0,
            "holdout_reference_content_accesses": 0,
        }

    sealed_receipt = {
        "schema": SEALED_RECEIPT_SCHEMA,
        "status": "complete",
        "split_seed": split_seed,
        "count": split_counts["holdout"],
        "ordered_identity_sha256": ordered_identity_sha256(split_identities["holdout"]),
        "identity_list_present": False,
        "question_answer_content_present": False,
        "content_materialized": False,
    }
    identity_receipts = {
        split: {
            "count": split_counts[split],
            "ordered_identity_sha256": ordered_identity_sha256(split_identities[split]),
        }
        for split in SPLITS
    }
    report = {
        "schema": FREEZE_REPORT_SCHEMA,
        "status": "complete",
        "split_seed": split_seed,
        "split_rule": "sha256(seed\\0identity)[:8] mod 10000; 0:7000 train, 7000:8500 development, 8500:10000 sealed",
        "counts": dict(split_counts),
        "task_counts": {split: dict(task_counts[split]) for split in NONSEALED_SPLITS},
        "identity_receipts": identity_receipts,
        "inputs": {
            "pairs_sha256": contract.pairs_sha256,
            "source_bank_sha256s": sorted(contract.bank_sha256s),
        },
        "draft_training_reference": {
            "corpus_sha256": contract.draft_training_sha256,
            "content_copied": False,
            "path_recorded": False,
            "hash_reference_only": True,
        },
        "revision_training_geometry": {
            "unique_train_identities": contract.split_counts["train"],
            "presentations": contract.revision_presentations,
            "single_correct_presentations_per_identity": 4,
            "other_presentations_per_identity": 1,
        },
        "input_schema_exception": {
            "historical_schema": LEGACY_SOURCE_SCHEMA,
            "hash_pinned_bank_sha256s": sorted(legacy_schema_bank_hashes),
            "paths_permitted": False,
            "emitted_assessor_schema": ASSESSOR_SCHEMA,
            "legacy_schema_emitted_to_model_views": False,
        },
        "input_directory_exception": {
            "historical_component": "product_reasoning",
            "hash_pinned_input_sha256s": sorted(historical_directory_hashes),
            "paths_emitted": False,
            "model_visible_paths_affected": False,
        },
        "outputs": {},
        "source_disjoint": True,
        "sealed_content_materialized": False,
        "protected_board_inputs": 0,
        "public_inputs": 0,
        "mbpp_reference_preflight": reference_preflight,
    }

    assessor_temporary = assessor_output.with_name(
        f".{assessor_output.name}.tmp.{os.getpid()}"
    )
    assessor_receipt_temporary = assessor_receipt_output.with_name(
        f".{assessor_receipt_output.name}.tmp.{os.getpid()}"
    )
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if (
        output.exists()
        or output.is_symlink()
        or assessor_output.exists()
        or assessor_output.is_symlink()
        or assessor_receipt_output.exists()
        or assessor_receipt_output.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
        or assessor_temporary.exists()
        or assessor_temporary.is_symlink()
        or assessor_receipt_temporary.exists()
        or assessor_receipt_temporary.is_symlink()
    ):
        raise PCF1DataError("refusing existing PCF1 source or assessor output")
    output.parent.mkdir(parents=True, exist_ok=True)
    assessor_output.parent.mkdir(parents=True, exist_ok=True)
    assessor_receipt_output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    assessor_published = False
    receipt_published = False
    source_published = False
    try:
        assessor_sha256 = _atomic_lines(assessor_temporary, confirmation_assessor_rows)
        assessor_receipt = {
            "schema": CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA,
            "status": "complete",
            "board_sha256": assessor_sha256,
            "rows": len(confirmation_assessor_rows),
            "semantic_access": "final_score_only",
        }
        assessor_receipt_sha256 = _atomic_json(
            assessor_receipt_temporary, assessor_receipt
        )
        receipts = {
            "train_sources.jsonl": {
                "sha256": _atomic_lines(temporary / "train_sources.jsonl", train_rows),
                "rows": len(train_rows),
            },
            "development_sources.jsonl": {
                "sha256": _atomic_lines(
                    temporary / "development_sources.jsonl", development_rows
                ),
                "rows": len(development_rows),
            },
            "confirmation_assessor_receipt": {
                "sha256": assessor_receipt_sha256,
                "board_sha256": assessor_sha256,
                "rows": 1,
            },
            "holdout_identity_receipt.json": {
                "sha256": _atomic_json(
                    temporary / "holdout_identity_receipt.json", sealed_receipt
                ),
                "rows": 1,
            },
        }
        if reference_sandbox_receipt is not None:
            reference_rows_sha256 = _atomic_lines(
                temporary / "mbpp_reference_preflight.jsonl",
                reference_preflight_rows,
            )
            if reference_rows_sha256 != reference_preflight["row_receipts_sha256"]:
                raise PCF1DataError("PCF1 reference-preflight serialization differs")
            receipts["mbpp_reference_preflight.jsonl"] = {
                "sha256": reference_rows_sha256,
                "rows": len(reference_preflight_rows),
            }
            reference_sandbox_sha256 = _atomic_json(
                temporary / "reference_sandbox_receipt.json",
                reference_sandbox_receipt,
            )
            receipts["reference_sandbox_receipt.json"] = {
                "sha256": reference_sandbox_sha256,
                "rows": 1,
            }
            reference_preflight["sandbox_receipt_sha256"] = reference_sandbox_sha256
        report["outputs"] = receipts
        _atomic_json(temporary / "report.json", report)
        try:
            os.link(assessor_temporary, assessor_output)
            assessor_published = True
            os.link(assessor_receipt_temporary, assessor_receipt_output)
            receipt_published = True
        except FileExistsError as error:
            raise PCF1DataError(
                "refusing existing PCF1 confirmation assessor output"
            ) from error
        os.rename(temporary, output)
        source_published = True
        for parent in {
            output.parent,
            assessor_output.parent,
            assessor_receipt_output.parent,
        }:
            parent_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        if source_published and output.exists():
            shutil.rmtree(output)
        if receipt_published:
            assessor_receipt_output.unlink(missing_ok=True)
        if assessor_published:
            assessor_output.unlink(missing_ok=True)
        raise
    finally:
        try:
            assessor_temporary.unlink()
        except FileNotFoundError:
            pass
        try:
            assessor_receipt_temporary.unlink()
        except FileNotFoundError:
            pass
    return report


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PCF1DataError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PCF1DataError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PCF1DataError(f"non-object {label}: {path}")
    return payload


def _load_source_view(
    path: Path, expected_schema: str, expected_split: str
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path, f"PCF1 {expected_split} source view"):
        _validate_schema(row.get("schema"), expected_schema, "PCF1 source view")
        identity = row.get("identity_sha256")
        if not _is_sha256(identity) or identity in rows:
            raise PCF1DataError("PCF1 source-view identity is invalid or duplicated")
        if row.get("split") != expected_split:
            raise PCF1DataError("PCF1 source-view split differs")
        if row.get("task") not in TASKS:
            raise PCF1DataError("PCF1 source-view task differs")
        if (
            not isinstance(row.get("source_prompt"), str)
            or not row["source_prompt"].strip()
            or row.get("runtime_fields") != ["source_prompt"]
        ):
            raise PCF1DataError("PCF1 source-view runtime boundary differs")
        assessor = row.get("assessor")
        if expected_split == "train":
            if (
                not isinstance(assessor, dict)
                or assessor.get("schema") != ASSESSOR_SCHEMA
                or assessor.get("identity_sha256") != identity
                or assessor.get("task") != row.get("task")
                or not isinstance(row.get("response"), str)
                or not row["response"].strip()
                or not isinstance(row.get("target_kind"), str)
                or not row["target_kind"]
            ):
                raise PCF1DataError("PCF1 training source supervision differs")
        elif any(
            field in row
            for field in (
                "assessor",
                "answer",
                "candidates",
                "correct",
                "gold",
                "response",
                "target",
            )
        ):
            raise PCF1DataError("PCF1 development source exposes supervision")
        rows[identity] = row
    return rows


_DRAFT_ALLOWED_FIELDS = frozenset(
    {
        "schema",
        "identity_sha256",
        "split",
        "task",
        "completion",
        "generated_tokens",
        "max_token_exhausted",
        "prompt_sha256",
        "adapter_checkpoint_sha256",
        "model_revision",
        "finish_reason",
        "wall_seconds",
    }
)


def _load_drafts(
    path: Path, sources: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    drafts: dict[str, dict[str, Any]] = {}
    adapter_hashes: set[str] = set()
    model_revisions: set[str] = set()
    forbidden_supervision = {
        "answer",
        "assessor",
        "candidates",
        "correct",
        "gold",
        "response",
        "target",
    }
    for row in _iter_jsonl(path, "PCF1 model drafts"):
        _validate_schema(row.get("schema"), DRAFT_SCHEMA, "PCF1 draft")
        if set(row) != _DRAFT_ALLOWED_FIELDS or forbidden_supervision & set(row):
            raise PCF1DataError("PCF1 draft row exposes an unauthorized field")
        identity = row.get("identity_sha256")
        if not _is_sha256(identity) or identity in drafts:
            raise PCF1DataError("PCF1 draft identity is invalid or duplicated")
        source = sources.get(identity)
        if source is None:
            raise PCF1DataError("PCF1 draft identity is not in the nonsealed universe")
        split = row.get("split")
        if split not in NONSEALED_SPLITS or split != source.get("split"):
            raise PCF1DataError("PCF1 draft requests a forbidden or mismatched split")
        if row.get("task") != source.get("task"):
            raise PCF1DataError("PCF1 draft task binding differs")
        completion = row.get("completion")
        if not isinstance(completion, str) or not completion.strip():
            raise PCF1DataError("PCF1 draft completion is empty")
        prompt_sha256 = row.get("prompt_sha256")
        expected_prompt_sha256 = hashlib.sha256(
            str(source["source_prompt"]).encode()
        ).hexdigest()
        adapter_sha256 = row.get("adapter_checkpoint_sha256")
        model_revision = row.get("model_revision")
        generated_tokens = row.get("generated_tokens")
        exhausted = row.get("max_token_exhausted")
        wall_seconds = row.get("wall_seconds")
        if (
            prompt_sha256 != expected_prompt_sha256
            or not _is_sha256(adapter_sha256)
            or model_revision != PINNED_MODEL_REVISION
            or isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or generated_tokens <= 0
            or not isinstance(exhausted, bool)
            or row.get("finish_reason") != ("length" if exhausted else "stop")
            or isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(float(wall_seconds))
            or wall_seconds < 0
        ):
            raise PCF1DataError("PCF1 draft provenance differs")
        adapter_hashes.add(adapter_sha256)
        model_revisions.add(model_revision)
        drafts[identity] = row
    if set(drafts) != set(sources):
        raise PCF1DataError("PCF1 draft coverage differs from the nonsealed universe")
    if len(adapter_hashes) != 1 or model_revisions != {PINNED_MODEL_REVISION}:
        raise PCF1DataError("PCF1 draft lineage is not uniform")
    return drafts


def materialize_drafts(
    *,
    source_root: Path,
    drafts_path: Path,
    assessor_receipt_path: Path,
    output: Path,
    contract: CustodyContract = FROZEN_CUSTODY,
) -> dict[str, Any]:
    """Bind exact nonsealed drafts to revision and commit/evaluation rows."""

    _validate_contract(contract, SPLIT_SEED)
    _validate_path(source_root, "source-root")
    _validate_path(drafts_path, "draft")
    _validate_path(assessor_receipt_path, "confirmation-assessor receipt")
    _validate_path(output, "output")
    report_path = source_root / "report.json"
    freeze_report = _load_json(report_path, "PCF1 freeze report")
    _validate_schema(
        freeze_report.get("schema"), FREEZE_REPORT_SCHEMA, "PCF1 freeze report"
    )
    if (
        freeze_report.get("status") != "complete"
        or freeze_report.get("split_seed") != SPLIT_SEED
    ):
        raise PCF1DataError("PCF1 freeze report is incomplete or uses another seed")
    if (
        freeze_report.get("counts") != dict(contract.split_counts)
        or freeze_report.get("inputs")
        != {
            "pairs_sha256": contract.pairs_sha256,
            "source_bank_sha256s": sorted(contract.bank_sha256s),
        }
        or freeze_report.get("draft_training_reference")
        != {
            "corpus_sha256": contract.draft_training_sha256,
            "content_copied": False,
            "path_recorded": False,
            "hash_reference_only": True,
        }
        or freeze_report.get("revision_training_geometry")
        != {
            "unique_train_identities": contract.split_counts["train"],
            "presentations": contract.revision_presentations,
            "single_correct_presentations_per_identity": 4,
            "other_presentations_per_identity": 1,
        }
        or freeze_report.get("source_disjoint") is not True
        or freeze_report.get("sealed_content_materialized") is not False
        or freeze_report.get("protected_board_inputs") != 0
        or freeze_report.get("public_inputs") != 0
    ):
        raise PCF1DataError("PCF1 freeze custody differs")
    schema_exception = freeze_report.get("input_schema_exception")
    if (
        not isinstance(schema_exception, dict)
        or schema_exception.get("historical_schema") != LEGACY_SOURCE_SCHEMA
        or not isinstance(schema_exception.get("hash_pinned_bank_sha256s"), list)
        or not set(schema_exception["hash_pinned_bank_sha256s"]).issubset(
            contract.bank_sha256s
        )
        or schema_exception.get("paths_permitted") is not False
        or schema_exception.get("emitted_assessor_schema") != ASSESSOR_SCHEMA
        or schema_exception.get("legacy_schema_emitted_to_model_views") is not False
    ):
        raise PCF1DataError("PCF1 historical schema exception differs")
    directory_exception = freeze_report.get("input_directory_exception")
    if (
        not isinstance(directory_exception, dict)
        or directory_exception.get("historical_component") != "product_reasoning"
        or not isinstance(directory_exception.get("hash_pinned_input_sha256s"), list)
        or not set(directory_exception["hash_pinned_input_sha256s"]).issubset(
            {contract.pairs_sha256, *contract.bank_sha256s}
        )
        or directory_exception.get("paths_emitted") is not False
        or directory_exception.get("model_visible_paths_affected") is not False
    ):
        raise PCF1DataError("PCF1 historical directory exception differs")
    outputs = freeze_report.get("outputs")
    if not isinstance(outputs, dict):
        raise PCF1DataError("PCF1 freeze output receipts are absent")
    filenames = {
        "train_sources.jsonl",
        "development_sources.jsonl",
        "holdout_identity_receipt.json",
    }
    expected_output_names = {*filenames, "confirmation_assessor_receipt"}
    if contract == FROZEN_CUSTODY:
        expected_output_names.update(
            {
                "mbpp_reference_preflight.jsonl",
                "reference_sandbox_receipt.json",
            }
        )
    if set(outputs) != expected_output_names:
        raise PCF1DataError("PCF1 freeze output names differ")
    for name in filenames:
        receipt = outputs.get(name)
        frozen_path = source_root / name
        if (
            not frozen_path.is_file()
            or not isinstance(receipt, dict)
            or receipt.get("sha256") != sha256_file(frozen_path)
            or not isinstance(receipt.get("rows"), int)
            or receipt.get("rows", 0) <= 0
        ):
            raise PCF1DataError(f"PCF1 frozen output hash differs: {name}")
    if contract == FROZEN_CUSTODY:
        from pcf1_code_sandbox import (
            CANDIDATE_POLICY_SHA256,
            SANDBOX_CONFIG_SHA256,
            PCF1SandboxError,
            validate_sandbox_receipt_payload,
        )

        reference_path = source_root / "reference_sandbox_receipt.json"
        reference_rows_path = source_root / "mbpp_reference_preflight.jsonl"
        reference_record = outputs.get("reference_sandbox_receipt.json")
        reference_rows_record = outputs.get("mbpp_reference_preflight.jsonl")
        reference_preflight = freeze_report.get("mbpp_reference_preflight")
        reference_receipt = _load_json(reference_path, "PCF1 reference sandbox receipt")
        try:
            validate_sandbox_receipt_payload(reference_receipt)
        except PCF1SandboxError as error:
            raise PCF1DataError("PCF1 reference sandbox receipt differs") from error
        expected_reference_rows = sum(
            int(freeze_report.get("task_counts", {}).get(split, {}).get("mbpp", 0))
            for split in NONSEALED_SPLITS
        )
        reference_rows = list(
            _iter_jsonl(reference_rows_path, "PCF1 MBPP reference preflight")
        )
        reference_identities = [
            str(row.get("identity_sha256")) for row in reference_rows
        ]
        setup_qualifications = sorted(
            {
                (
                    str(row.get("setup_source_sha256")),
                    str(row.get("setup_qualification_sha256")),
                )
                for row in reference_rows
            }
        )
        setup_qualification_lines = b"".join(
            (
                json.dumps(
                    {
                        "setup_source_sha256": setup_sha256,
                        "setup_qualification_sha256": qualification_sha256,
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode()
            for setup_sha256, qualification_sha256 in setup_qualifications
        )
        if (
            not isinstance(reference_record, dict)
            or reference_record != {"sha256": sha256_file(reference_path), "rows": 1}
            or reference_rows_record
            != {
                "sha256": sha256_file(reference_rows_path),
                "rows": expected_reference_rows,
            }
            or len(reference_rows) != expected_reference_rows
            or reference_identities != sorted(reference_identities)
            or len(set(reference_identities)) != expected_reference_rows
            or any(
                set(row)
                != {
                    "identity_sha256",
                    "split",
                    "candidate_source_sha256",
                    "program_sha256",
                    "setup_source_sha256",
                    "setup_qualification_sha256",
                    "candidate_policy_sha256",
                    "sandbox_config_sha256",
                    "allocation_probe_sha256",
                    "termination_classification",
                }
                or row.get("split") not in NONSEALED_SPLITS
                or any(
                    not _is_sha256(row.get(field))
                    for field in (
                        "identity_sha256",
                        "candidate_source_sha256",
                        "program_sha256",
                        "setup_source_sha256",
                        "setup_qualification_sha256",
                    )
                )
                or row.get("candidate_policy_sha256") != CANDIDATE_POLICY_SHA256
                or row.get("sandbox_config_sha256") != SANDBOX_CONFIG_SHA256
                or row.get("allocation_probe_sha256")
                != reference_receipt.get("probe_sha256")
                or row.get("termination_classification") != "trusted_tests_completed"
                for row in reference_rows
            )
            or not isinstance(reference_preflight, dict)
            or reference_preflight.get("schema")
            != "shohin-pcf1-mbpp-reference-preflight-v1"
            or reference_preflight.get("status") != "pass"
            or reference_preflight.get("scope") != ["train", "development"]
            or reference_preflight.get("rows") != expected_reference_rows
            or reference_preflight.get("ordered_identity_sha256")
            != ordered_identity_sha256(reference_identities)
            or reference_preflight.get("row_receipts_sha256")
            != sha256_file(reference_rows_path)
            or reference_preflight.get("unique_setups") != len(setup_qualifications)
            or reference_preflight.get("setup_pair_receipts_sha256")
            != hashlib.sha256(setup_qualification_lines).hexdigest()
            or reference_preflight.get("candidate_policy_sha256")
            != CANDIDATE_POLICY_SHA256
            or reference_preflight.get("sandbox_config_sha256") != SANDBOX_CONFIG_SHA256
            or reference_preflight.get("allocation_probe_sha256")
            != reference_receipt.get("probe_sha256")
            or reference_preflight.get("sandbox_receipt_sha256")
            != sha256_file(reference_path)
            or reference_preflight.get("all_policy_accepted") is not True
            or reference_preflight.get("all_sandbox_passed") is not True
            or reference_preflight.get("holdout_reference_content_accesses") != 0
        ):
            raise PCF1DataError("PCF1 MBPP reference preflight differs")
    assessor_receipt_record = outputs.get("confirmation_assessor_receipt")
    assessor_receipt = _load_json(
        assessor_receipt_path, "PCF1 confirmation assessor receipt"
    )
    if (
        not isinstance(assessor_receipt_record, dict)
        or assessor_receipt_record.get("sha256") != sha256_file(assessor_receipt_path)
        or assessor_receipt.get("schema") != CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA
        or assessor_receipt.get("status") != "complete"
        or not _is_sha256(assessor_receipt.get("board_sha256"))
        or assessor_receipt.get("board_sha256")
        != assessor_receipt_record.get("board_sha256")
        or assessor_receipt.get("rows") != contract.split_counts["development"]
        or assessor_receipt.get("semantic_access") != "final_score_only"
    ):
        raise PCF1DataError("PCF1 confirmation assessor receipt differs")

    sealed_receipt = _load_json(
        source_root / "holdout_identity_receipt.json", "PCF1 sealed identity receipt"
    )
    if (
        sealed_receipt.get("schema") != SEALED_RECEIPT_SCHEMA
        or sealed_receipt.get("status") != "complete"
        or not isinstance(sealed_receipt.get("count"), int)
        or sealed_receipt.get("count", 0) <= 0
        or not _is_sha256(sealed_receipt.get("ordered_identity_sha256"))
        or sealed_receipt.get("identity_list_present") is not False
        or sealed_receipt.get("question_answer_content_present") is not False
        or sealed_receipt.get("content_materialized") is not False
    ):
        raise PCF1DataError("PCF1 sealed identity receipt differs")
    identity_receipts = freeze_report.get("identity_receipts")
    if (
        not isinstance(identity_receipts, dict)
        or identity_receipts.get("holdout")
        != {
            "count": sealed_receipt["count"],
            "ordered_identity_sha256": sealed_receipt["ordered_identity_sha256"],
        }
        or sealed_receipt["count"] != contract.split_counts["holdout"]
    ):
        raise PCF1DataError("PCF1 sealed identity binding differs")

    train_sources = _load_source_view(
        source_root / "train_sources.jsonl", TRAIN_SOURCE_SCHEMA, "train"
    )
    development_sources = _load_source_view(
        source_root / "development_sources.jsonl",
        DEVELOPMENT_SOURCE_SCHEMA,
        "development",
    )
    if set(train_sources) & set(development_sources):
        raise PCF1DataError("PCF1 nonsealed source views overlap")
    for split, rows in (
        ("train", train_sources),
        ("development", development_sources),
    ):
        expected_receipt = {
            "count": len(rows),
            "ordered_identity_sha256": ordered_identity_sha256(rows),
        }
        if (
            identity_receipts.get(split) != expected_receipt
            or len(rows) != contract.split_counts[split]
        ):
            raise PCF1DataError(f"PCF1 {split} identity binding differs")
    sources = {**train_sources, **development_sources}
    drafts = _load_drafts(drafts_path, sources)

    revision_rows: list[dict[str, Any]] = []
    commit_rows: list[dict[str, Any]] = []
    development_rows: list[dict[str, Any]] = []
    for identity in sorted(sources):
        source = sources[identity]
        draft = drafts[identity]
        task = str(source["task"])
        source_prompt = str(source["source_prompt"])
        completion = str(draft["completion"]).strip()
        treatment_prompt = revision_prompt(source_prompt, completion)
        draft_sha256 = hashlib.sha256(completion.encode()).hexdigest()
        common = {
            "identity_sha256": identity,
            "task": task,
            "question": treatment_prompt,
            "source_prompt": source_prompt,
            "internal_draft": {**draft, "completion": completion},
            "candidates": [],
            "runtime_fields": ["question", "source_prompt"],
            "internal_draft_visible": True,
            "external_candidate_text_visible": False,
        }
        if source["split"] == "train":
            common["assessor"] = source["assessor"]
            common["runtime_fields"] = ["question"]
            outcome_class = source.get("outcome_class")
            for presentation in range(revision_presentations(outcome_class)):
                revision_rows.append(
                    {
                        "schema": REVISION_TRAIN_SCHEMA,
                        "identity_sha256": hashlib.sha256(
                            f"pcf1-revision\0{identity}\0{presentation}".encode()
                        ).hexdigest(),
                        "source_identity_sha256": identity,
                        "task": task,
                        "outcome_class": outcome_class,
                        "presentation": presentation,
                        "question": treatment_prompt,
                        "model_owned_draft_sha256": draft_sha256,
                        "response": source["response"],
                        "target_kind": source["target_kind"],
                        "runtime_fields": ["question"],
                        "internal_draft_visible": True,
                        "external_candidate_text_visible": False,
                        "supervisor_only_fields": [
                            "response",
                            "target_kind",
                            "task",
                            "outcome_class",
                        ],
                    }
                )
            commit_rows.append(
                {
                    **common,
                    "schema": EVAL_SCHEMA,
                    "split": "calibration",
                }
            )
        else:
            development_rows.append(
                {
                    **common,
                    "schema": EVAL_SCHEMA,
                    "split": "confirmation",
                }
            )

    if (
        len(revision_rows) != contract.revision_presentations
        or len(commit_rows) != len(train_sources)
        or len(development_rows) != len(development_sources)
    ):
        raise PCF1DataError("PCF1 materialized row geometry differs")
    materialization_report = {
        "schema": MATERIALIZATION_REPORT_SCHEMA,
        "status": "complete",
        "split_seed": SPLIT_SEED,
        "inputs": {
            "freeze_report_sha256": sha256_file(report_path),
            "drafts_sha256": sha256_file(drafts_path),
            "draft_rows": len(drafts),
        },
        "identity_receipts": {
            "train": {
                "count": len(train_sources),
                "ordered_identity_sha256": ordered_identity_sha256(train_sources),
            },
            "development": {
                "count": len(development_sources),
                "ordered_identity_sha256": ordered_identity_sha256(development_sources),
            },
            "sealed": {
                "count": sealed_receipt["count"],
                "ordered_identity_sha256": sealed_receipt["ordered_identity_sha256"],
                "content_materialized": False,
            },
        },
        "counts": {
            "train_unique_identities": len(train_sources),
            "revision_train_presentations": len(revision_rows),
            "calibration_rows": len(commit_rows),
            "confirmation_rows": len(development_rows),
        },
        "revision_presentation_rule": {
            "single_correct": 4,
            "both_correct_or_both_wrong": 1,
        },
        "outputs": {},
        "confirmation_assessor_access": {
            "semantic_reads": 0,
            "authorized_reader": "score_pcf1_commit.py",
        },
        "source_disjoint": True,
        "sealed_content_materialized": False,
        "protected_board_inputs": 0,
        "public_inputs": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }

    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if (
        output.exists()
        or output.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise PCF1DataError(f"refusing existing PCF1 output root: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        receipts = {
            "revision_train": {
                "path": str((output / "revision_train.jsonl").resolve()),
                "sha256": _atomic_lines(
                    temporary / "revision_train.jsonl", revision_rows
                ),
                "rows": len(revision_rows),
            },
            "calibration": {
                "path": str((output / "commit_train_eval.jsonl").resolve()),
                "sha256": _atomic_lines(
                    temporary / "commit_train_eval.jsonl", commit_rows
                ),
                "rows": len(commit_rows),
            },
            "confirmation": {
                "path": str((output / "development_eval.jsonl").resolve()),
                "sha256": _atomic_lines(
                    temporary / "development_eval.jsonl", development_rows
                ),
                "rows": len(development_rows),
            },
            "confirmation_assessors": {
                "sha256": assessor_receipt["board_sha256"],
                "rows": assessor_receipt["rows"],
                "semantic_access": "final_score_only",
            },
            "confirmation_assessor_receipt": {
                "sha256": sha256_file(assessor_receipt_path),
                "rows": 1,
            },
        }
        materialization_report["outputs"] = receipts
        _atomic_json(temporary / "materialization_report.json", materialization_report)
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return materialization_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze nonsealed CVG1 sources")
    freeze.add_argument("--pairs", type=Path, required=True)
    freeze.add_argument(
        "--bank", dest="banks", type=Path, action="append", required=True
    )
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--assessor-output", type=Path, required=True)
    freeze.add_argument("--assessor-receipt-output", type=Path, required=True)
    freeze.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    materialize = subparsers.add_parser(
        "materialize", help="bind exact nonsealed model drafts"
    )
    materialize.add_argument("--source-root", type=Path, required=True)
    materialize.add_argument("--drafts", type=Path, required=True)
    materialize.add_argument("--assessor-receipt", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "freeze":
        from pcf1_code_sandbox import (
            qualify_allocation,
            preflight_mbpp_reference,
            preflight_mbpp_setup,
        )

        reference_sandbox_receipt = qualify_allocation()
        setup_qualifications: dict[str, dict[str, Any]] = {}

        def evaluate_reference(source: dict[str, Any], split: str) -> dict[str, Any]:
            setup = str(source.get("test_setup_code") or "")
            setup_sha256 = hashlib.sha256(setup.encode()).hexdigest()
            if setup_sha256 not in setup_qualifications:
                setup_qualifications[setup_sha256] = preflight_mbpp_setup(setup)
            return preflight_mbpp_reference(
                source,
                split=split,
                setup_qualification=setup_qualifications[setup_sha256],
            )

        report = freeze_sources(
            pairs_path=args.pairs,
            bank_paths=args.banks,
            output=args.output,
            assessor_output=args.assessor_output,
            assessor_receipt_output=args.assessor_receipt_output,
            reference_evaluator=evaluate_reference,
            reference_sandbox_receipt=reference_sandbox_receipt,
            split_seed=args.split_seed,
        )
    else:
        report = materialize_drafts(
            source_root=args.source_root,
            drafts_path=args.drafts,
            assessor_receipt_path=args.assessor_receipt,
            output=args.output,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
