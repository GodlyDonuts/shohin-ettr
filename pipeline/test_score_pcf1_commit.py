"""Tests for the one-open PCF1 confirmation scoring custodian."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

import score_pcf1_commit as scorer
from build_pcf1_commit_pairs import PCF1PairError, load_arm
from score_pcf1_commit import (
    AUTHORIZATION_SCHEMA,
    OUTCOME_SCHEMA,
    PCF1ScoreError,
    PINNED_MODEL_REVISION,
    _authorization_hashes,
    load_selections,
    publish_score_root,
    score,
    validate_confirmation_pairs,
)
from pcf1_code_sandbox import CANDIDATE_POLICY_SHA256

TOTAL = 1289
MODEL_ROOT = "/immutable/models/ministral"
ADAPTER_SHA256 = "2" * 64
COMMIT_SHA256 = "3" * 64
SEALED_ZERO = {"holdout": 0, "product": 0, "public": 0}


def _sandbox_receipt() -> dict[str, Any]:
    return {
        "schema": "shohin-pcf1-code-sandbox-receipt-v1",
        "status": "pass",
        "sandbox_config_sha256": scorer.SANDBOX_CONFIG_SHA256,
        "bwrap_sha256": scorer.BWRAP_SHA256,
        "probe_sha256": "8" * 64,
        "sandbox_isolation_passed": True,
    }


def _sandbox_receipt_sha256() -> str:
    encoded = (json.dumps(_sandbox_receipt(), indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


@pytest.fixture(autouse=True)
def _qualified_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scorer,
        "qualify_allocation",
        _sandbox_receipt,
    )
    monkeypatch.setattr(
        scorer,
        "validate_environment_receipt",
        lambda *_: {"environment_tree": {"sha256": "9" * 64}},
    )
    setup_receipt = {
        "schema": "shohin-pcf1-mbpp-setup-qualification-v1",
        "status": "pass",
        "setup_source_sha256": hashlib.sha256(b"").hexdigest(),
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "sandbox_config_sha256": scorer.SANDBOX_CONFIG_SHA256,
        "allocation_probe_sha256": "8" * 64,
        "termination_classification": "trusted_tests_completed",
    }
    setup_receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(setup_receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(
        scorer,
        "qualify_mbpp_assessor_setups",
        lambda _assessors: [setup_receipt],
    )


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_digest(identities: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(identities)) + "\n").encode()).hexdigest()


Mutation = Callable[[dict[str, dict[str, Any]]], None]


def _fixture(
    root: Path,
    mutation: Mutation | None = None,
    *,
    empty_revision: bool = False,
) -> tuple[argparse.Namespace, dict[str, dict[str, Any]]]:
    tasks = ("math500", "bbh_logic", "mbpp")
    sources: list[dict[str, Any]] = []
    assessors: list[dict[str, Any]] = []
    candidates: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ("revision", "unchanged", "self_refinement")
    }
    selections: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    identities: list[str] = []
    for index in range(TOTAL):
        identity = hashlib.sha256(f"pcf1-score-{index}".encode()).hexdigest()
        identities.append(identity)
        task = tasks[index % len(tasks)]
        draft = f"draft {identity}"
        source_prompt = f"source question {index}"
        sources.append(
            {
                "schema": "shohin-pcf1-eval-v1",
                "identity_sha256": identity,
                "split": "confirmation",
                "task": task,
                "question": f"{source_prompt}\n{draft}",
                "source_prompt": source_prompt,
                "runtime_fields": ["question", "source_prompt"],
                "internal_draft_visible": True,
                "external_candidate_text_visible": False,
                "internal_draft": {
                    "identity_sha256": identity,
                    "completion": draft,
                },
                "candidates": [],
            }
        )
        assessors.append(
            {
                "schema": "shohin-pcf1-confirmation-assessor-v1",
                "identity_sha256": identity,
                "split": "confirmation",
                "task": task,
                "assessor": {"identity_sha256": identity, "task": task},
            }
        )
        correctness = {
            "revision": index % 2 == 0,
            "unchanged": index % 3 == 0,
            "self_refinement": index % 5 == 0,
        }
        for arm in candidates:
            completion = f"{arm}:{'correct' if correctness[arm] else 'wrong'}"
            if empty_revision and arm == "revision" and index == 0:
                completion = ""
            candidates[arm].append(
                {
                    "schema": "shohin-pcf1-candidate-v1",
                    "arm": arm,
                    "identity_sha256": identity,
                    "task": task,
                    "completion": completion,
                    "generated_tokens": 8,
                    "max_token_exhausted": False,
                }
            )
        selections.append(
            {
                "schema": "shohin-pcf1-commit-selection-v1",
                "identity_sha256": identity,
                "task": task,
                "selected_index": 0,
                "selected_lineage": "revision",
                "order_consistent": True,
                "margin": 1.0,
            }
        )
        pairs.append(
            {
                "schema": "shohin-pcf1-confirmation-pair-v1",
                "identity_sha256": identity,
                "split": "confirmation",
                "task": task,
                "question": f"{source_prompt}\n{draft}",
                "candidates": [
                    {
                        "lineage": "revision",
                        "completion": candidates["revision"][-1]["completion"],
                    },
                    {
                        "lineage": "unchanged",
                        "completion": candidates["unchanged"][-1]["completion"],
                    },
                ],
            }
        )

    data = _write_jsonl(root / "development_eval.jsonl", sources)
    assessor_board = _write_jsonl(root / "confirmation_assessors.jsonl", assessors)
    assessor_receipt = _write_json(
        root / "confirmation_assessor_receipt.json",
        {
            "schema": "shohin-pcf1-confirmation-assessor-receipt-v1",
            "status": "complete",
            "board_sha256": _sha256(assessor_board),
            "rows": TOTAL,
            "semantic_access": "final_score_only",
        },
    )
    candidates_root = root / "candidates"
    candidates_root.mkdir()
    candidate_paths = {
        arm: _write_jsonl(candidates_root / f"{arm}.jsonl", rows)
        for arm, rows in candidates.items()
    }
    selections_path = _write_jsonl(root / "selections.jsonl", selections)
    pairs_path = _write_jsonl(root / "pairs.jsonl", pairs)
    reports: dict[str, dict[str, Any]] = {
        "data": {
            "schema": "shohin-pcf1-data-report-v1",
            "status": "complete",
            "outputs": {
                "confirmation": {
                    "path": str(data),
                    "sha256": _sha256(data),
                    "rows": TOTAL,
                },
                "confirmation_assessors": {
                    "sha256": _sha256(assessor_board),
                    "rows": TOTAL,
                    "semantic_access": "final_score_only",
                },
                "confirmation_assessor_receipt": {
                    "sha256": _sha256(assessor_receipt),
                    "rows": 1,
                },
            },
            "confirmation_assessor_access": {
                "semantic_reads": 0,
                "authorized_reader": "score_pcf1_commit.py",
            },
            "sealed_access": SEALED_ZERO,
        },
        "pairs": {
            "schema": "shohin-pcf1-confirmation-pair-report-v1",
            "status": "complete",
            "rows": TOTAL,
            "labels_or_correctness_fields": 0,
            "source_disjoint_from_calibration": True,
            "output": str(pairs_path),
            "output_sha256": _sha256(pairs_path),
            "inputs": {},
            "sealed_access": SEALED_ZERO,
        },
        "training": {
            "schema": "shohin-pcf1-commit-training-report-v1",
            "status": "complete",
            "model_root": MODEL_ROOT,
            "model_revision": PINNED_MODEL_REVISION,
            "model_loader": "multimodal",
            "adapter_checkpoint_sha256": ADAPTER_SHA256,
            "checkpoint_sha256": COMMIT_SHA256,
            "protected_adapter_sha256_after": ADAPTER_SHA256,
            "protected_adapter_unchanged": True,
            "updates": 128,
            "gradient_accumulation": 8,
            "head_width": 512,
            "max_sequence_length": 3072,
            "seed": 2026080822,
            "backbone_learning_rate": 2e-6,
            "head_learning_rate": 2e-4,
            "sealed_access": SEALED_ZERO,
        },
        "application": {
            "schema": "shohin-pcf1-commit-application-report-v1",
            "status": "complete",
            "model_root": MODEL_ROOT,
            "model_revision": PINNED_MODEL_REVISION,
            "model_loader": "multimodal",
            "adapter_checkpoint_sha256": ADAPTER_SHA256,
            "protected_adapter_unchanged": True,
            "commit_checkpoint_sha256": COMMIT_SHA256,
            "pairs_sha256": _sha256(pairs_path),
            "selections": str(selections_path),
            "selections_sha256": _sha256(selections_path),
            "rows": TOTAL,
            "prompt_truncated": 0,
            "malformed": 0,
            "order_consistent": TOTAL,
            "maximum_swap_error": 0.0,
            "max_sequence_length": 3072,
            "correctness_or_task_label_visible": False,
            "sealed_access": SEALED_ZERO,
        },
    }
    for arm in candidates:
        reports[arm] = {
            "schema": "shohin-pcf1-merged-evaluation-v1",
            "status": "complete",
            "arm": arm,
            "split": "confirmation",
            "assessment_mode": "confirmation_deferred",
            "assessor_board_access_count": 0,
            "exact_identity_coverage": True,
            "candidates_output": str(candidate_paths[arm]),
            "candidates_sha256": _sha256(candidate_paths[arm]),
            "data": str(data),
            "data_sha256": _sha256(data),
            "metrics": None,
            "sealed_access": SEALED_ZERO,
        }
    if mutation is not None:
        mutation(reports)

    report_paths = {name: root / f"{name}_report.json" for name in reports}
    for arm in candidates:
        _write_json(report_paths[arm], reports[arm])
    reports["pairs"]["inputs"] = {
        "data_sha256": _sha256(data),
        "revision_report_sha256": _sha256(report_paths["revision"]),
        "unchanged_report_sha256": _sha256(report_paths["unchanged"]),
        "revision_candidates_sha256": _sha256(candidate_paths["revision"]),
        "unchanged_candidates_sha256": _sha256(candidate_paths["unchanged"]),
    }
    _write_json(report_paths["data"], reports["data"])
    _write_json(report_paths["pairs"], reports["pairs"])
    reports["application"]["pairs_report_sha256"] = _sha256(report_paths["pairs"])
    _write_json(report_paths["training"], reports["training"])
    _write_json(report_paths["application"], reports["application"])

    auxiliary = {}
    for name in (
        "mechanics",
        "data_custody",
        "model_custody",
        "runtime_custody",
        "prescore_dispatch",
        "prescore_accounting",
    ):
        auxiliary[name] = root / f"{name}.json"
        auxiliary[name].write_text(f"{name}\n", encoding="utf-8")
    authorization_path = root / "score_authorization.json"
    environment_receipt = root / "environment_receipt.json"
    environment_receipt.write_text("environment\n", encoding="utf-8")
    args = argparse.Namespace(
        confirmation_data=data,
        confirmation_assessors=assessor_board,
        confirmation_assessor_receipt=assessor_receipt,
        data_report=report_paths["data"],
        revision_report=report_paths["revision"],
        revision_candidates=candidate_paths["revision"],
        unchanged_report=report_paths["unchanged"],
        unchanged_candidates=candidate_paths["unchanged"],
        self_refinement_report=report_paths["self_refinement"],
        self_refinement_candidates=candidate_paths["self_refinement"],
        candidates_root=candidates_root,
        confirmation_pairs=pairs_path,
        confirmation_pairs_report=report_paths["pairs"],
        selections=selections_path,
        application_report=report_paths["application"],
        training_report=report_paths["training"],
        mechanics_report=auxiliary["mechanics"],
        data_custody=auxiliary["data_custody"],
        model_custody=auxiliary["model_custody"],
        runtime_custody=auxiliary["runtime_custody"],
        prescore_dispatch_receipt=auxiliary["prescore_dispatch"],
        prescore_accounting_receipt=auxiliary["prescore_accounting"],
        prescore_authorization=authorization_path,
        output_root=root / "score_result",
        sandbox_probe_output=root / "score_result.sandbox-probe.json",
        environment_receipt=environment_receipt,
        environment_receipt_sha256=_sha256(environment_receipt),
    )
    authorization = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "complete",
        "run_id": "pcf1-test-run",
        "scoring_authorized": True,
        "one_shot": True,
        "rows": TOTAL,
        "identity_order_sha256": _identity_digest(identities),
        "confirmation_assessors_sha256": _sha256(assessor_board),
        "score_output_root": str(args.output_root.resolve()),
        "environment_tree_sha256": "9" * 64,
        "code_sandbox_config_sha256": scorer.SANDBOX_CONFIG_SHA256,
        "code_sandbox_binary_sha256": scorer.BWRAP_SHA256,
        "code_sandbox_probe_sha256": _sandbox_receipt_sha256(),
        "code_sandbox_probe_result_sha256": "8" * 64,
        "code_sandbox_receipt_sha256": _sandbox_receipt_sha256(),
        "assessor_board_access_count_before": 0,
        "sealed_access": SEALED_ZERO,
        **_authorization_hashes(args),
    }
    _write_json(authorization_path, authorization)
    return args, reports


def _fake_score(_: dict[str, Any], completion: str) -> dict[str, bool]:
    return {"correct": completion.endswith(":correct")}


def test_scores_all_arms_in_one_authorized_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    calls = 0
    assessor_opens = 0
    original_open = Path.open

    def tracked_open(path: Path, *open_args: Any, **open_kwargs: Any) -> Any:
        nonlocal assessor_opens
        if path.resolve() == args.confirmation_assessors.resolve():
            assessor_opens += 1
        return original_open(path, *open_args, **open_kwargs)

    def counted(assessor: dict[str, Any], completion: str) -> dict[str, bool]:
        nonlocal calls
        calls += 1
        return _fake_score(assessor, completion)

    monkeypatch.setattr(scorer, "score_completion", counted)
    monkeypatch.setattr(Path, "open", tracked_open)
    result = score(args)
    assert calls == TOTAL * 3
    assert assessor_opens == 1
    assert result["assessment_calls"] == TOTAL * 3
    assert result["assessor_board_semantic_reads"] == 1
    assert result["confirmation_open_count"] == 1
    assert result["authorization_consumed"] is True
    assert result["score_consumption_state"] == "consumed"
    consumption = Path(result["score_consumption"])
    assert consumption.is_file()
    assert result["score_consumption_sha256"] == _sha256(consumption)
    assert result["confirmation"]["overall"]["revision_correct"] == (TOTAL + 1) // 2
    assert (
        result["arm_metrics"]["self_refinement"]["overall"]["generated_correct"]
        == (TOTAL + 4) // 5
    )
    assert (args.output_root / "report.json").is_file()
    outcomes = (args.output_root / "outcomes.jsonl").read_text().splitlines()
    assert len(outcomes) == TOTAL
    first_outcome = json.loads(outcomes[0])
    assert first_outcome["schema"] == OUTCOME_SCHEMA
    assert (
        first_outcome["score_consumption_sha256"] == result["score_consumption_sha256"]
    )
    assert result["outcomes_sha256"] == _sha256(args.output_root / "outcomes.jsonl")


def test_empty_completion_is_one_shot_malformed_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path, empty_revision=True)
    calls = 0

    def counted(assessor: dict[str, Any], completion: str) -> dict[str, bool]:
        nonlocal calls
        calls += 1
        return _fake_score(assessor, completion)

    monkeypatch.setattr(scorer, "score_completion", counted)
    result = score(args)
    assert result["assessment_calls"] == TOTAL * 3
    assert result["score_completion_calls"] == TOTAL * 3 - 1
    assert result["arm_malformed"] == {
        "revision": 1,
        "unchanged": 0,
        "self_refinement": 0,
    }
    assert result["confirmation_malformed_candidates"] == 1
    assert result["confirmation_malformed_selections"] == 1
    assert calls == TOTAL * 3 - 1
    first = json.loads(
        (args.output_root / "outcomes.jsonl").read_text().splitlines()[0]
    )
    assert first["revision_malformed"] is True
    assert first["revision_correct"] is False
    assert first["selected_malformed"] is True


def test_capability_policy_rejection_is_explicit_malformed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    first_call = True

    def policy_once(_: dict[str, Any], completion: str) -> dict[str, Any]:
        nonlocal first_call
        if first_call:
            first_call = False
            return {
                "correct": False,
                "execution": {
                    "candidate_policy_passed": False,
                    "termination_classification": "candidate_policy_rejection",
                },
            }
        return {"correct": completion.endswith(":correct"), "execution": None}

    monkeypatch.setattr(scorer, "score_completion", policy_once)
    result = score(args)
    first = json.loads(
        (args.output_root / "outcomes.jsonl").read_text().splitlines()[0]
    )
    assert result["arm_capability_policy_rejected"] == {
        "revision": 1,
        "unchanged": 0,
        "self_refinement": 0,
    }
    assert result["confirmation_capability_policy_rejections"] == 1
    assert result["confirmation_malformed_candidates"] == 1
    assert first["revision_capability_policy_rejected"] is True
    assert first["revision_malformed"] is True


def test_rejects_authorization_before_semantic_assessor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    authorization = json.loads(args.prescore_authorization.read_text())
    authorization["arm_candidates_sha256s"]["revision"] = "0" * 64
    _write_json(args.prescore_authorization, authorization)
    called = False

    def forbidden(*_: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("assessor board must remain unopened")

    monkeypatch.setattr(scorer, "load_assessors_once", forbidden)
    with pytest.raises(PCF1ScoreError, match="authorization binding"):
        score(args)
    assert called is False
    assert not args.output_root.exists()


def test_rejects_scored_confirmation_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    rows = [
        json.loads(line) for line in args.revision_candidates.read_text().splitlines()
    ]
    rows[0]["correct"] = True
    _write_jsonl(args.revision_candidates, rows)
    monkeypatch.setattr(scorer, "score_completion", _fake_score)
    with pytest.raises(PCF1ScoreError, match="label-free revision arm"):
        score(args)
    assert not args.output_root.exists()


def test_pair_loader_rejects_explicit_candidate_outside_safe_root(
    tmp_path: Path,
) -> None:
    args, _ = _fixture(tmp_path)
    narrow_root = tmp_path / "narrow_candidates"
    narrow_root.mkdir()

    with pytest.raises(PCF1PairError, match="escapes the explicit root"):
        load_arm(
            args.revision_report,
            args.revision_candidates,
            "revision",
            "confirmation",
            candidates_root=narrow_root,
        )


def test_score_rejects_candidate_outside_explicit_root_before_board_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    narrow_root = tmp_path / "narrow_candidates"
    narrow_root.mkdir()
    args.candidates_root = narrow_root
    called = False

    def forbidden(*_: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("assessor board must remain unopened")

    monkeypatch.setattr(scorer, "load_assessors_once", forbidden)
    with pytest.raises(PCF1ScoreError, match="label-free revision arm"):
        score(args)
    assert called is False


def test_refuses_overwriting_atomic_score_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    monkeypatch.setattr(scorer, "score_completion", _fake_score)
    score(args)
    with pytest.raises(PCF1ScoreError, match="existing PCF1 result"):
        score(args)


def test_authorization_rejects_a_second_output_root_before_assessor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    args.output_root = tmp_path / "different_score_result"
    called = False

    def forbidden(*_: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("assessor board must remain unopened")

    monkeypatch.setattr(scorer, "load_assessors_once", forbidden)
    with pytest.raises(PCF1ScoreError, match="authorization binding"):
        score(args)
    assert called is False
    assert not args.output_root.exists()
    assert not scorer.score_consumption_path(args.output_root).exists()


def test_post_claim_failure_permanently_consumes_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    calls = 0

    def fail_after_claim(*_: Any) -> Any:
        nonlocal calls
        calls += 1
        raise PCF1ScoreError("injected post-claim assessor failure")

    monkeypatch.setattr(scorer, "load_assessors_once", fail_after_claim)
    with pytest.raises(PCF1ScoreError, match="injected post-claim"):
        score(args)
    claim = scorer.score_consumption_path(args.output_root)
    assert claim.is_file()
    payload = json.loads(claim.read_text(encoding="utf-8"))
    assert payload["claim_state"] == "consumed"
    assert payload["score_output_root"] == str(args.output_root.resolve())
    assert calls == 1
    assert not args.output_root.exists()
    failure = scorer.score_terminal_failure_path(args.output_root)
    assert failure.is_file()
    terminal = json.loads(failure.read_text(encoding="utf-8"))
    assert terminal["schema"] == scorer.TERMINAL_FAILURE_SCHEMA
    assert terminal["status"] == "infrastructure_failure"
    assert terminal["failure_phase"] == "assessor_board_read"
    assert terminal["assessor_bytes_read"] == 0
    assert terminal["retry_authorized"] is False
    assert terminal["successor_authorized"] is False

    with pytest.raises(PCF1ScoreError, match="already consumed"):
        score(args)
    assert calls == 1


def test_rejects_selection_index_lineage_tamper(tmp_path: Path) -> None:
    args, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in args.selections.read_text().splitlines()]
    rows[0]["selected_lineage"] = "unchanged"
    _write_jsonl(args.selections, rows)
    with pytest.raises(PCF1ScoreError, match="selection content"):
        load_selections(args.selections)


def test_rejects_pair_completion_or_order_tamper(tmp_path: Path) -> None:
    identity = "a" * 64
    source = {
        "identity_sha256": identity,
        "task": "math500",
        "question": "question",
    }
    candidates = {
        "revision": {identity: {"completion": "revision"}},
        "unchanged": {identity: {"completion": "unchanged"}},
    }
    pair_path = _write_jsonl(
        tmp_path / "pairs.jsonl",
        [
            {
                "schema": "shohin-pcf1-confirmation-pair-v1",
                "identity_sha256": identity,
                "split": "confirmation",
                "task": "math500",
                "question": "question",
                "candidates": [
                    {"lineage": "revision", "completion": "unchanged"},
                    {"lineage": "unchanged", "completion": "revision"},
                ],
            }
        ],
    )
    with pytest.raises(PCF1ScoreError, match="pair/arm binding"):
        validate_confirmation_pairs(pair_path, [source], candidates)


def test_atomic_score_commit_failure_leaves_no_partial_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "score_root"

    def fail_rename(_: Path, __: Path) -> None:
        raise OSError("injected score commit failure")

    monkeypatch.setattr(scorer.os, "rename", fail_rename)
    with pytest.raises(OSError, match="injected score commit failure"):
        publish_score_root(
            output_root,
            [{"schema": OUTCOME_SCHEMA, "identity_sha256": "a" * 64}],
            {"schema": "test", "status": "complete"},
        )
    assert not output_root.exists()
    assert not any(tmp_path.glob(".score_root.tmp.*"))
