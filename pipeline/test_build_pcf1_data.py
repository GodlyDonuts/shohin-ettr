from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

import build_pcf1_data as pcf1_data
from build_pcf1_data import (
    ASSESSOR_SCHEMA,
    DEVELOPMENT_SOURCE_SCHEMA,
    DRAFT_SCHEMA,
    EVAL_SCHEMA,
    FROZEN_CUSTODY,
    FREEZE_REPORT_SCHEMA,
    MATERIALIZATION_REPORT_SCHEMA,
    PINNED_MODEL_REVISION,
    PAIR_SCHEMA,
    PCF1DataError,
    REVISION_TRAIN_SCHEMA,
    SEALED_RECEIPT_SCHEMA,
    SPLIT_SEED,
    LEGACY_SOURCE_SCHEMA,
    CustodyContract,
    assigned_split,
    freeze_sources,
    materialize_drafts,
    ordered_identity_sha256,
    revision_presentations,
    revision_prompt,
    sha256_file,
)


def test_revision_prompt_is_task_agnostic_and_uses_source_format_contract() -> None:
    expected = revision_prompt("original with its own format", "draft")
    assert expected == (
        "Solve the original problem by checking and revising the model's earlier draft. "
        "The draft may contain useful steps or errors; do not merely critique it.\n\n"
        "Original problem:\noriginal with its own format\n\nInternal draft:\ndraft\n\n"
        "Follow the original problem's requested output format.\n\n"
        "Original problem:\noriginal with its own format"
    )
    assert "mbpp" not in revision_prompt.__code__.co_varnames


def test_materialization_api_has_no_assessor_board_input() -> None:
    assert "assessor_path" not in inspect.signature(materialize_drafts).parameters


def test_frozen_freeze_contract_requires_reference_preflight(tmp_path: Path) -> None:
    with pytest.raises(PCF1DataError, match="requires MBPP reference preflight"):
        freeze_sources(
            pairs_path=tmp_path / "missing-pairs",
            bank_paths=[tmp_path / "missing-bank"],
            output=tmp_path / "safe",
            assessor_output=tmp_path / "assessors.jsonl",
            assessor_receipt_output=tmp_path / "assessor-receipt.json",
        )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    (
        ("base_only", 4),
        ("expert_only", 4),
        ("both_correct", 1),
        ("both_wrong", 1),
    ),
)
def test_revision_presentation_geometry(outcome: str, expected: int) -> None:
    assert revision_presentations(outcome) == expected
    assert FROZEN_CUSTODY.revision_presentations == 9655


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _identities(counts: dict[str, int]) -> dict[str, list[str]]:
    result = {split: [] for split in counts}
    index = 0
    while any(len(result[split]) < counts[split] for split in counts):
        identity = hashlib.sha256(f"pcf1-fixture-{index}".encode()).hexdigest()
        split = assigned_split(identity)
        if len(result[split]) < counts[split]:
            result[split].append(identity)
        index += 1
    return result


def _fixture(tmp_path: Path) -> dict[str, Any]:
    split_counts = {"train": 3, "development": 3, "holdout": 2}
    identities = _identities(split_counts)
    task_cycle = ("math500", "bbh_logic", "mbpp")
    assignment: dict[str, tuple[str, str]] = {}
    for split, values in identities.items():
        for index, identity in enumerate(values):
            assignment[identity] = (split, task_cycle[index % len(task_cycle)])

    pair_rows: list[dict[str, Any]] = []
    bank_rows: dict[str, list[dict[str, Any]]] = {task: [] for task in task_cycle}
    for identity, (split, task) in sorted(assignment.items()):
        is_sealed = split == "holdout"
        marker = f"SEALED_SECRET_{identity}" if is_sealed else f"question {identity}"
        if task == "mbpp":
            source = {
                "schema": "shohin-cvg1-mbpp-rollout-bank-v1",
                "identity_sha256": identity,
                "task": task,
                "text": marker,
                "code": f"def f():\n    return '{marker}'",
                "test_list": ["assert f()"],
                "test_setup_code": "",
                "reference_execution_sha256": hashlib.sha256(
                    identity.encode()
                ).hexdigest(),
            }
            question = marker
        else:
            source = {
                "schema": f"shohin-cvg1-{task}-rollout-bank-v1",
                "identity_sha256": identity,
                "task": task,
                "question": marker,
                "answer": f"answer for {marker}",
            }
            question = marker
        bank_rows[task].append(source)
        pair_rows.append(
            {
                "schema": PAIR_SCHEMA,
                "identity_sha256": identity,
                "split": "legacy_partition_is_ignored",
                "task": task,
                "question": question,
                "outcome_class": "base_only",
                "candidates": [
                    {
                        "lineage": "base",
                        "completion": f"verified completion {identity}",
                        "correct": True,
                    },
                    {
                        "lineage": "expert",
                        "completion": f"incorrect completion {identity}",
                        "correct": False,
                    },
                ],
            }
        )

    pairs_path = tmp_path / "pairs.jsonl"
    _write_jsonl(pairs_path, pair_rows)
    bank_paths: list[Path] = []
    for index, task in enumerate(task_cycle):
        path = tmp_path / f"bank_{index}.jsonl"
        _write_jsonl(path, bank_rows[task])
        bank_paths.append(path)
    contract = CustodyContract(
        pairs_sha256=sha256_file(pairs_path),
        bank_sha256s=frozenset(sha256_file(path) for path in bank_paths),
        draft_training_sha256="d" * 64,
        split_seed=SPLIT_SEED,
        split_counts=split_counts,
        revision_presentations=12,
    )
    return {
        "pairs": pairs_path,
        "banks": bank_paths,
        "contract": contract,
        "identities": identities,
        "assignment": assignment,
    }


def _freeze(tmp_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    fixture = _fixture(tmp_path)
    output = tmp_path / "frozen"
    report = freeze_sources(
        pairs_path=fixture["pairs"],
        bank_paths=fixture["banks"],
        output=output,
        assessor_output=tmp_path / "confirmation_assessors.jsonl",
        assessor_receipt_output=tmp_path / "confirmation_assessor_receipt.json",
        contract=fixture["contract"],
    )
    return fixture, output, report


def test_freeze_binds_nonsealed_mbpp_reference_preflight_without_holdout_access(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    seen: list[tuple[str, str]] = []

    def evaluate(source: dict[str, Any], split: str) -> dict[str, Any]:
        seen.append((source["identity_sha256"], split))
        return {
            "identity_sha256": source["identity_sha256"],
            "split": split,
            "candidate_source_sha256": hashlib.sha256(
                source["code"].encode()
            ).hexdigest(),
            "program_sha256": hashlib.sha256(
                (source["code"] + "\n" + "\n".join(source["test_list"])).encode()
            ).hexdigest(),
            "setup_source_sha256": hashlib.sha256(b"").hexdigest(),
            "setup_qualification_sha256": "e" * 64,
            "candidate_policy_sha256": "a" * 64,
            "sandbox_config_sha256": "b" * 64,
            "allocation_probe_sha256": "c" * 64,
            "reference_assessment_mode": "trusted_reference",
            "generated_candidate_policy_applied": False,
            "termination_classification": "trusted_tests_completed",
        }

    output = tmp_path / "frozen"
    report = freeze_sources(
        pairs_path=fixture["pairs"],
        bank_paths=fixture["banks"],
        output=output,
        assessor_output=tmp_path / "confirmation_assessors.jsonl",
        assessor_receipt_output=tmp_path / "confirmation_assessor_receipt.json",
        reference_evaluator=evaluate,
        reference_sandbox_receipt={"schema": "fixture", "status": "pass"},
        contract=fixture["contract"],
    )

    expected = sorted(
        identity
        for identity, (split, task) in fixture["assignment"].items()
        if split != "holdout" and task == "mbpp"
    )
    assert sorted(identity for identity, _ in seen) == expected
    assert all(split in {"train", "development"} for _, split in seen)
    preflight = report["mbpp_reference_preflight"]
    assert preflight["status"] == "pass"
    assert preflight["rows"] == len(expected)
    assert preflight["ordered_identity_sha256"] == ordered_identity_sha256(expected)
    assert preflight["holdout_reference_content_accesses"] == 0
    assert preflight["unique_setups"] == 1
    assert preflight["reference_assessment_mode"] == "trusted_reference"
    assert preflight["generated_candidate_policy_applied"] is False
    assert preflight["all_references_passed"] is True
    assert preflight["sandbox_receipt_sha256"] == sha256_file(
        output / "reference_sandbox_receipt.json"
    )
    assert preflight["row_receipts_sha256"] == sha256_file(
        output / "mbpp_reference_preflight.jsonl"
    )


def test_materialize_accepts_exact_trusted_reference_preflight_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pcf1_code_sandbox import CANDIDATE_POLICY_SHA256, SANDBOX_CONFIG_SHA256
    import pcf1_code_sandbox

    fixture = _fixture(tmp_path)
    monkeypatch.setattr(pcf1_data, "FROZEN_CUSTODY", fixture["contract"])
    monkeypatch.setattr(
        pcf1_code_sandbox, "validate_sandbox_receipt_payload", lambda _receipt: None
    )

    def evaluate(source: dict[str, Any], split: str) -> dict[str, Any]:
        candidate = source["code"]
        program = candidate + "\n" + "\n".join(source["test_list"])
        return {
            "identity_sha256": source["identity_sha256"],
            "split": split,
            "candidate_source_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
            "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
            "setup_source_sha256": hashlib.sha256(b"").hexdigest(),
            "setup_qualification_sha256": "e" * 64,
            "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
            "sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
            "allocation_probe_sha256": "c" * 64,
            "reference_assessment_mode": "trusted_reference",
            "generated_candidate_policy_applied": False,
            "termination_classification": "trusted_tests_completed",
        }

    source_root = tmp_path / "frozen"
    freeze_sources(
        pairs_path=fixture["pairs"],
        bank_paths=fixture["banks"],
        output=source_root,
        assessor_output=tmp_path / "confirmation_assessors.jsonl",
        assessor_receipt_output=tmp_path / "confirmation_assessor_receipt.json",
        reference_evaluator=evaluate,
        reference_sandbox_receipt={"probe_sha256": "c" * 64},
        contract=fixture["contract"],
    )
    report = materialize_drafts(
        source_root=source_root,
        drafts_path=_drafts(fixture, tmp_path / "drafts.jsonl"),
        assessor_receipt_path=tmp_path / "confirmation_assessor_receipt.json",
        output=tmp_path / "materialized",
        contract=fixture["contract"],
    )

    assert report["status"] == "complete"
    assert report["sealed_access"] == {"holdout": 0, "product": 0, "public": 0}


def _drafts(fixture: dict[str, Any], path: Path, *, extra: str | None = None) -> Path:
    rows = []
    for split in ("train", "development"):
        for identity in fixture["identities"][split]:
            _, task = fixture["assignment"][identity]
            source = next(
                row
                for row in _rows(
                    fixture["banks"][("math500", "bbh_logic", "mbpp").index(task)]
                )
                if row["identity_sha256"] == identity
            )
            prompt = (
                source["question"]
                if task != "mbpp"
                else "Write Python code that solves the task and passes every test. "
                "Return only executable Python code, without Markdown fences.\n\n"
                f"Task:\n{source['text']}\n\nTests:\n" + "\n".join(source["test_list"])
            )
            if task == "bbh_logic":
                prompt = (
                    f"{source['question']}\n\nReason carefully, then put only the exact "
                    "requested answer or option label inside \\boxed{}."
                )
            rows.append(
                {
                    "schema": DRAFT_SCHEMA,
                    "identity_sha256": identity,
                    "split": split,
                    "task": task,
                    "completion": f"model draft {identity}",
                    "generated_tokens": 12,
                    "max_token_exhausted": False,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "adapter_checkpoint_sha256": "a" * 64,
                    "model_revision": PINNED_MODEL_REVISION,
                    "finish_reason": "stop",
                    "wall_seconds": 0.1,
                }
            )
    if extra is not None:
        _, task = fixture["assignment"][extra]
        rows.append(
            {
                "schema": DRAFT_SCHEMA,
                "identity_sha256": extra,
                "split": "holdout",
                "task": task,
                "completion": "must never be admitted",
                "generated_tokens": 1,
                "max_token_exhausted": False,
                "prompt_sha256": "0" * 64,
                "adapter_checkpoint_sha256": "a" * 64,
                "model_revision": PINNED_MODEL_REVISION,
                "finish_reason": "stop",
                "wall_seconds": 0.1,
            }
        )
    _write_jsonl(path, rows)
    return path


def test_freeze_writes_nonsealed_views_and_content_free_receipt(tmp_path: Path) -> None:
    fixture, output, report = _freeze(tmp_path)
    assert report["schema"] == FREEZE_REPORT_SCHEMA
    assert set(path.name for path in output.iterdir()) == {
        "train_sources.jsonl",
        "development_sources.jsonl",
        "holdout_identity_receipt.json",
        "report.json",
    }
    train = _rows(output / "train_sources.jsonl")
    development = _rows(output / "development_sources.jsonl")
    assert {row["split"] for row in train} == {"train"}
    assert {row["split"] for row in development} == {"development"}
    assert {row["schema"] for row in development} == {DEVELOPMENT_SOURCE_SCHEMA}
    assert {row["assessor"]["schema"] for row in train} == {ASSESSOR_SCHEMA}
    assert all("assessor" not in row for row in development)
    assessor_path = tmp_path / "confirmation_assessors.jsonl"
    assert assessor_path.parent == tmp_path and assessor_path.parent != output
    assessors = _rows(assessor_path)
    assert len(assessors) == len(development)
    assert {row["identity_sha256"] for row in assessors} == {
        row["identity_sha256"] for row in development
    }
    assert {row["identity_sha256"] for row in train}.isdisjoint(
        row["identity_sha256"] for row in development
    )
    assessor_receipt = json.loads(
        (tmp_path / "confirmation_assessor_receipt.json").read_text()
    )
    assert set(assessor_receipt) == {
        "schema",
        "status",
        "board_sha256",
        "rows",
        "semantic_access",
    }
    assert "board" not in assessor_receipt

    receipt = json.loads((output / "holdout_identity_receipt.json").read_text())
    assert receipt == {
        "schema": SEALED_RECEIPT_SCHEMA,
        "status": "complete",
        "split_seed": SPLIT_SEED,
        "count": 2,
        "ordered_identity_sha256": ordered_identity_sha256(
            fixture["identities"]["holdout"]
        ),
        "identity_list_present": False,
        "question_answer_content_present": False,
        "content_materialized": False,
    }
    emitted = b"".join(path.read_bytes() for path in output.iterdir())
    assert b"SEALED_SECRET" not in emitted
    for identity in fixture["identities"]["holdout"]:
        assert identity.encode() not in emitted
    assert report["draft_training_reference"] == {
        "corpus_sha256": "d" * 64,
        "content_copied": False,
        "path_recorded": False,
        "hash_reference_only": True,
    }
    assert set(report["task_counts"]) == {"train", "development"}
    assert report["input_schema_exception"] == {
        "historical_schema": LEGACY_SOURCE_SCHEMA,
        "hash_pinned_bank_sha256s": [],
        "paths_permitted": False,
        "emitted_assessor_schema": ASSESSOR_SCHEMA,
        "legacy_schema_emitted_to_model_views": False,
    }
    assert report["input_directory_exception"] == {
        "historical_component": "product_reasoning",
        "hash_pinned_input_sha256s": [],
        "paths_emitted": False,
        "model_visible_paths_affected": False,
    }


def test_materialize_binds_one_eval_row_per_nonsealed_source(tmp_path: Path) -> None:
    fixture, source_root, _ = _freeze(tmp_path)
    drafts = _drafts(fixture, tmp_path / "drafts.jsonl")
    output = tmp_path / "materialized"
    report = materialize_drafts(
        source_root=source_root,
        drafts_path=drafts,
        assessor_receipt_path=tmp_path / "confirmation_assessor_receipt.json",
        output=output,
        contract=fixture["contract"],
    )
    assert report["schema"] == MATERIALIZATION_REPORT_SCHEMA
    assert set(report["outputs"]) == {
        "revision_train",
        "calibration",
        "confirmation",
        "confirmation_assessors",
        "confirmation_assessor_receipt",
    }
    assert report["sealed_access"] == {"holdout": 0, "product": 0, "public": 0}

    revision = _rows(output / "revision_train.jsonl")
    calibration = _rows(output / "commit_train_eval.jsonl")
    confirmation = _rows(output / "development_eval.jsonl")
    assert len(revision) == 12
    assert len(calibration) == 3
    assert len(confirmation) == 3
    assert {row["schema"] for row in revision} == {REVISION_TRAIN_SCHEMA}
    assert {row["presentation"] for row in revision} == {0, 1, 2, 3}
    assert report["counts"] == {
        "train_unique_identities": 3,
        "revision_train_presentations": 12,
        "calibration_rows": 3,
        "confirmation_rows": 3,
    }
    assert {row["split"] for row in calibration} == {"calibration"}
    assert {row["split"] for row in confirmation} == {"confirmation"}
    for row in calibration + confirmation:
        assert row["schema"] == EVAL_SCHEMA
        assert row["internal_draft_visible"] is True
        assert row["external_candidate_text_visible"] is False
        assert row["candidates"] == []
        assert row["internal_draft"]["completion"] in row["question"]
    assert all(row["runtime_fields"] == ["question"] for row in calibration)
    assert all(row["assessor"]["task"] == row["task"] for row in calibration)
    assert all(
        row["runtime_fields"] == ["question", "source_prompt"] and "assessor" not in row
        for row in confirmation
    )

    emitted = b"".join(path.read_bytes() for path in output.iterdir())
    assert b"SEALED_SECRET" not in emitted
    for identity in fixture["identities"]["holdout"]:
        assert identity.encode() not in emitted


def test_rejects_draft_outside_nonsealed_universe(tmp_path: Path) -> None:
    fixture, source_root, _ = _freeze(tmp_path)
    extra = fixture["identities"]["holdout"][0]
    drafts = _drafts(fixture, tmp_path / "drafts_extra.jsonl", extra=extra)
    with pytest.raises(PCF1DataError, match="nonsealed universe"):
        materialize_drafts(
            source_root=source_root,
            drafts_path=drafts,
            assessor_receipt_path=tmp_path / "confirmation_assessor_receipt.json",
            output=tmp_path / "materialized",
            contract=fixture["contract"],
        )


@pytest.mark.parametrize("term", ("holdout", "product", "public"))
def test_path_firewall_rejects_protected_board_destination(
    tmp_path: Path, term: str
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(PCF1DataError, match=f"containing {term}"):
        freeze_sources(
            pairs_path=fixture["pairs"],
            bank_paths=fixture["banks"],
            output=tmp_path / f"{term}_board",
            assessor_output=tmp_path / "confirmation_assessors.jsonl",
            assessor_receipt_output=tmp_path / "confirmation_assessor_receipt.json",
            contract=fixture["contract"],
        )


def test_freeze_rejects_changed_seed_and_custody(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(PCF1DataError, match="split seed differs"):
        freeze_sources(
            pairs_path=fixture["pairs"],
            bank_paths=fixture["banks"],
            output=tmp_path / "frozen",
            assessor_output=tmp_path / "confirmation_assessors.jsonl",
            assessor_receipt_output=tmp_path / "confirmation_assessor_receipt.json",
            split_seed=SPLIT_SEED + 1,
            contract=fixture["contract"],
        )
    wrong = CustodyContract(
        pairs_sha256="0" * 64,
        bank_sha256s=fixture["contract"].bank_sha256s,
        draft_training_sha256="d" * 64,
        split_seed=SPLIT_SEED,
        split_counts=fixture["contract"].split_counts,
        revision_presentations=fixture["contract"].revision_presentations,
    )
    with pytest.raises(PCF1DataError, match="pair SHA-256 differs"):
        freeze_sources(
            pairs_path=fixture["pairs"],
            bank_paths=fixture["banks"],
            output=tmp_path / "frozen",
            assessor_output=tmp_path / "confirmation_assessors.jsonl",
            assessor_receipt_output=tmp_path / "confirmation_assessor_receipt.json",
            contract=wrong,
        )


def test_freeze_commit_failure_removes_uncommitted_assessor_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "frozen"
    assessor = tmp_path / "confirmation_assessors.jsonl"
    receipt = tmp_path / "confirmation_assessor_receipt.json"

    def fail_rename(_: Path, __: Path) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(pcf1_data.os, "rename", fail_rename)
    with pytest.raises(OSError, match="injected publication failure"):
        freeze_sources(
            pairs_path=fixture["pairs"],
            bank_paths=fixture["banks"],
            output=output,
            assessor_output=assessor,
            assessor_receipt_output=receipt,
            contract=fixture["contract"],
        )
    assert not output.exists()
    assert not assessor.exists()
    assert not receipt.exists()


def test_freeze_rejects_duplicate_source_identity(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    duplicate = _rows(fixture["banks"][0])[0]
    rows = _rows(fixture["banks"][1]) + [duplicate]
    _write_jsonl(fixture["banks"][1], rows)
    contract = CustodyContract(
        pairs_sha256=sha256_file(fixture["pairs"]),
        bank_sha256s=frozenset(sha256_file(path) for path in fixture["banks"]),
        draft_training_sha256="d" * 64,
        split_seed=SPLIT_SEED,
        split_counts=fixture["contract"].split_counts,
        revision_presentations=fixture["contract"].revision_presentations,
    )
    with pytest.raises(PCF1DataError, match="duplicated"):
        freeze_sources(
            pairs_path=fixture["pairs"],
            bank_paths=fixture["banks"],
            output=tmp_path / "frozen",
            assessor_output=tmp_path / "confirmation_assessors.jsonl",
            assessor_receipt_output=tmp_path / "confirmation_assessor_receipt.json",
            contract=contract,
        )


def test_historical_schema_is_hash_bound_and_sanitized(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    legacy_path = fixture["banks"][0]
    legacy_rows = _rows(legacy_path)
    for row in legacy_rows:
        row["schema"] = LEGACY_SOURCE_SCHEMA
    _write_jsonl(legacy_path, legacy_rows)
    contract = CustodyContract(
        pairs_sha256=sha256_file(fixture["pairs"]),
        bank_sha256s=frozenset(sha256_file(path) for path in fixture["banks"]),
        draft_training_sha256="d" * 64,
        split_seed=SPLIT_SEED,
        split_counts=fixture["contract"].split_counts,
        revision_presentations=fixture["contract"].revision_presentations,
    )
    output = tmp_path / "frozen"
    report = freeze_sources(
        pairs_path=fixture["pairs"],
        bank_paths=fixture["banks"],
        output=output,
        assessor_output=tmp_path / "confirmation_assessors.jsonl",
        assessor_receipt_output=tmp_path / "confirmation_assessor_receipt.json",
        contract=contract,
    )
    assert report["input_schema_exception"]["hash_pinned_bank_sha256s"] == [
        sha256_file(legacy_path)
    ]
    model_views = (output / "train_sources.jsonl").read_bytes() + (
        output / "development_sources.jsonl"
    ).read_bytes()
    assert LEGACY_SOURCE_SCHEMA.encode() not in model_views
    assert ASSESSOR_SCHEMA.encode() in model_views


def test_historical_directory_is_hash_bound_and_not_reemitted(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    historical = tmp_path / "product_reasoning"
    historical.mkdir()
    moved_pairs = historical / "pairs.jsonl"
    fixture["pairs"].rename(moved_pairs)
    fixture["pairs"] = moved_pairs
    output = tmp_path / "frozen"
    report = freeze_sources(
        pairs_path=fixture["pairs"],
        bank_paths=fixture["banks"],
        output=output,
        assessor_output=tmp_path / "confirmation_assessors.jsonl",
        assessor_receipt_output=tmp_path / "confirmation_assessor_receipt.json",
        contract=fixture["contract"],
    )
    assert report["input_directory_exception"]["hash_pinned_input_sha256s"] == [
        fixture["contract"].pairs_sha256
    ]
    assert all("product" not in name for name in report["outputs"])
