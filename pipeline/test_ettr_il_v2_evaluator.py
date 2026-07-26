from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import ettr_il_v2_evaluator as evaluator
from ettr_il_v2_evaluator import (
    ARMS,
    CHARGED_FLOPS,
    COMPLETE_SYSTEM_PARAMETERS,
    CONTROLS,
    ENCODED_TOKENS,
    EXPECTED_ENDPOINTS,
    EXPECTED_RUNS,
    FOLDS,
    ONTOLOGIES,
    OPEN_CONFIRMATION,
    PANEL_SCHEMA,
    POSITIVE_DECISION,
    PROTOCOL,
    QUERY_ROW_EXPOSURES,
    STRATA,
    TRAINABLE_PARAMETERS,
    EvaluationError,
    aggregate_panel,
    canonical_json_bytes,
    evaluate_immutable_panel,
    parse_panel,
)
from ettr_il_v2_schedule import MODEL_SEEDS


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _count(numerator: int = 1000, denominator: int = 1000) -> dict[str, int]:
    return {"denominator": denominator, "numerator": numerator}


def _diagnostics() -> dict[str, object]:
    values: dict[str, object] = {
        name: _count()
        for name in evaluator._SCALAR_DIAGNOSTICS
    }
    values["learned_initial_packet_exact"] = [
        {"denominator": 1000, "numerator": 1000, "ontology": ontology}
        for ontology in ONTOLOGIES
    ]
    return values


def _cell_cores(
    *,
    arm: str,
    fold: int,
    ontology: str,
    stratum: str,
) -> list[dict[str, object]]:
    count = 24 if stratum == "all_axes" else 32
    views = 4 if stratum == "all_axes" else 3
    if arm == "treatment":
        exact = 4 * views
    elif arm == "dense_state" and stratum == "seen_id":
        exact = 11
    elif stratum == "all_axes":
        exact = 10
    else:
        exact = 8
    return [
        {
            "exact_causal_rectangles": exact,
            "semantic_core_id": _digest(
                f"core|{fold}|{ontology}|{stratum}|{index}"
            ),
            "view_count": views,
        }
        for index in range(count)
    ]


def _run(arm: str, fold: int, seed: int) -> dict[str, object]:
    same_group = f"{fold}|{seed}"
    return {
        "arm": arm,
        "arm_config_sha256": _digest(f"arm-config|{arm}"),
        "budget_receipt_sha256": _digest(f"budget|{arm}|{same_group}"),
        "charged_flops": CHARGED_FLOPS,
        "checkpoint_sha256": _digest(f"checkpoint|{arm}|{same_group}"),
        "checkpoint_update": 6000,
        "complete_system_parameters": COMPLETE_SYSTEM_PARAMETERS,
        "control_receipt_sha256": _digest(f"control|{arm}|{same_group}"),
        "custody_receipt_sha256": _digest(f"custody|{arm}|{same_group}"),
        "dataset_sha256": _digest(f"dataset|{fold}"),
        "diagnostics": _diagnostics() if arm == "treatment" else {},
        "encoded_tokens": ENCODED_TOKENS,
        "endpoint_receipt_sha256": _digest(f"endpoint|{arm}|{same_group}"),
        "fit_semantic_core_exact": [
            {
                "denominator": 288,
                "numerator": 288,
                "ontology": ontology,
            }
            for ontology in evaluator.FIT_ONTOLOGIES[fold]
        ],
        "fold": fold,
        "objective_weights_sha256": _digest("objective-weights"),
        "optimizer_receipt_sha256": _digest(f"optimizer|{arm}|{same_group}"),
        "optimizer_updates": 6000,
        "query_row_exposures": QUERY_ROW_EXPOSURES,
        "schedule_sha256": _digest(f"schedule|{same_group}"),
        "seed": seed,
        "source_payload_sha256": _digest(f"source|{fold}"),
        "static_flop_receipt_sha256": _digest(f"flops|{arm}|{same_group}"),
        "static_loss_path_flops": 987_654_321,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "cells": [
            {
                "cores": _cell_cores(
                    arm=arm,
                    fold=fold,
                    ontology=ontology,
                    stratum=stratum,
                ),
                "ontology": ontology,
                "stratum": stratum,
            }
            for ontology in ONTOLOGIES
            for stratum in STRATA
        ],
    }


def _panel(split: str = "development") -> dict[str, object]:
    return {
        "dataset_root_sha256": _digest("dataset-root"),
        "endpoint_root_sha256": _digest("endpoint-root"),
        "protocol": PROTOCOL,
        "runs": [
            _run(arm, fold, seed)
            for arm in ARMS
            for fold in FOLDS
            for seed in MODEL_SEEDS
        ],
        "schema": PANEL_SCHEMA,
        "split": split,
        "split_plaintext_sha256": _digest(f"{split}-plaintext"),
    }


def _fast_passing_lcbs(
    panel: evaluator.ParsedPanel,
    observed: tuple[evaluator.Fraction, ...],
) -> tuple[float, ...]:
    del panel
    return tuple(float(value) - 0.01 for value in observed)


def _write_immutable(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)
    path.chmod(0o444)
    return hashlib.sha256(payload).hexdigest()


def _authorization(
    panel_sha256: str,
    evaluator_root_sha256: str,
) -> dict[str, object]:
    return {
        "authorization_nonce": "12" * 32,
        "authorizer_signature": "34" * 64,
        "checkpoint_selection_allowed": False,
        "confirmation_envelope_roots": [
            _digest(f"confirmation-envelope|{fold}") for fold in FOLDS
        ],
        "dataset_root_sha256": _digest("dataset-root"),
        "development_result_sha256": _digest("development-result"),
        "endpoint_root_sha256": _digest("endpoint-root"),
        "evaluator_root_sha256": evaluator_root_sha256,
        "expires_at_utc": "2099-01-01T00:00:00Z",
        "panel_sha256": panel_sha256,
        "protocol": PROTOCOL,
        "rescore_allowed": False,
        "retry_allowed": False,
        "schema": evaluator.OPEN_AUTHORIZATION_SCHEMA,
    }


def test_gold_panel_has_exact_locked_coverage_and_pairing() -> None:
    parsed = parse_panel(_panel())
    assert parsed.split == "development"
    assert len(parsed.runs) == EXPECTED_RUNS == 75
    assert {
        (run.arm, run.fold, run.seed) for run in parsed.runs.values()
    } == {
        (arm, fold, seed)
        for arm in ARMS
        for fold in FOLDS
        for seed in MODEL_SEEDS
    }


def test_gold_development_opens_confirmation_with_292_adjusted_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "_compute_simultaneous_lcbs",
        _fast_passing_lcbs,
    )
    result = aggregate_panel(parse_panel(_panel()))
    assert result["all_gates_pass"] is True
    assert result["decision"] == OPEN_CONFIRMATION
    assert result["primary_localization"] == "none"
    assert len(result["statistical_endpoints"]) == EXPECTED_ENDPOINTS == 292
    assert result["bootstrap"]["method"] == (
        "hierarchical_paired_single_step_max_deviation"
    )
    assert result["bootstrap"]["replicates"] == 100_000
    assert {
        endpoint["control"] for endpoint in result["statistical_endpoints"]
    } == set(CONTROLS)


def test_observed_and_bootstrap_effects_are_paired_by_global_seed_and_core() -> None:
    parsed = parse_panel(_panel())
    order, observed = evaluator._observed_effects(parsed)
    assert order[0] == ("state_reset", None)
    assert observed[0] == evaluator.Fraction(65, 192)

    first_replicate = next(iter(evaluator._bootstrap_effects(parsed)))
    plan = evaluator.statistics.build_bootstrap_plan(
        parsed.split_plaintext_sha256,
        0,
        (cell.bootstrap_cell for cell in evaluator._all_cells()),
    )
    sampled = {
        evaluator.CellKey(cell.fold, cell.ontology, cell.stratum): indices
        for cell, indices in plan.cell_indices
    }
    cell_effects: list[float] = []
    for cell in evaluator._all_cells():
        values = []
        for seed_index in plan.model_seed_indices:
            seed = MODEL_SEEDS[seed_index]
            treatment = parsed.runs[
                ("treatment", cell.fold, seed)
            ].cells[cell]
            control = parsed.runs[
                ("state_reset", cell.fold, seed)
            ].cells[cell]
            values.extend(
                float(treatment[index].rate - control[index].rate)
                for index in sampled[cell]
            )
        cell_effects.append(sum(values) / len(values))
    assert first_replicate[0] == pytest.approx(
        sum(cell_effects) / len(cell_effects)
    )


def test_update_6000_is_the_only_admitted_checkpoint() -> None:
    panel = _panel()
    panel["runs"][0]["checkpoint_update"] = 3000
    with pytest.raises(EvaluationError, match="update-6000-only"):
        parse_panel(panel)
    panel = _panel()
    panel["runs"][0]["optimizer_updates"] = 5999
    with pytest.raises(EvaluationError, match="update-6000-only"):
        parse_panel(panel)


def test_missing_seed_arm_or_extra_endpoint_is_rejected() -> None:
    panel = _panel()
    panel["runs"].pop()
    with pytest.raises(EvaluationError, match="run count"):
        parse_panel(panel)

    panel = _panel()
    panel["runs"][-1] = deepcopy(panel["runs"][0])
    with pytest.raises(EvaluationError, match="duplicated"):
        parse_panel(panel)

    panel = _panel()
    panel["runs"][0]["cells"][0]["stratum"] = "post_hoc_favorable"
    with pytest.raises(EvaluationError, match="identity"):
        parse_panel(panel)


def test_unpaired_core_or_unequal_budget_is_rejected() -> None:
    panel = _panel()
    panel["runs"][15]["cells"][0]["cores"][0]["semantic_core_id"] = _digest(
        "substituted-core"
    )
    with pytest.raises(EvaluationError, match="paired semantic-core"):
        parse_panel(panel)

    panel = _panel()
    panel["runs"][1]["static_loss_path_flops"] += 1
    with pytest.raises(EvaluationError, match="static loss-path FLOP"):
        parse_panel(panel)


def test_absolute_failure_closes_and_localizes_query_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "_compute_simultaneous_lcbs",
        _fast_passing_lcbs,
    )
    panel = _panel()
    for run in panel["runs"]:
        if run["arm"] == "treatment":
            run["diagnostics"]["gold_terminal"] = _count(994, 1000)
    result = aggregate_panel(parse_panel(panel))
    assert result["all_gates_pass"] is False
    assert result["decision"] == evaluator.CLOSE_WITHOUT_CONFIRMATION
    assert result["primary_localization"] == "query_reader"


def test_fit_gate_is_exact_fold_level_576_core_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "_compute_simultaneous_lcbs",
        _fast_passing_lcbs,
    )
    panel = _panel()
    treatment = next(
        run
        for run in panel["runs"]
        if run["arm"] == "treatment"
        and run["fold"] == 0
        and run["seed"] == MODEL_SEEDS[0]
    )
    treatment["fit_semantic_core_exact"][0]["numerator"] = 283
    result = aggregate_panel(parse_panel(panel))
    gate = next(
        value
        for value in result["absolute_gates"]
        if value["gate"]
        == f"fold=0|seed={MODEL_SEEDS[0]}|fit_semantic_core_exact"
    )
    assert gate["observed"] == "571/576"
    assert gate["passed"] is True


def test_gold_initial_failure_localizes_reactor_after_reader_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "_compute_simultaneous_lcbs",
        _fast_passing_lcbs,
    )
    panel = _panel()
    for run in panel["runs"]:
        if run["arm"] == "treatment":
            run["diagnostics"]["gold_initial"] = _count(989, 1000)
    result = aggregate_panel(parse_panel(panel))
    assert result["primary_localization"] == "reactor_transaction_interface"


def test_multiplicity_adjusted_lcb_is_noncompensatory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_bounds(
        panel: evaluator.ParsedPanel,
        observed: tuple[evaluator.Fraction, ...],
    ) -> tuple[float, ...]:
        del panel
        bounds = [float(value) - 0.01 for value in observed]
        bounds[0] = 0.10
        return tuple(bounds)

    monkeypatch.setattr(
        evaluator,
        "_compute_simultaneous_lcbs",
        failing_bounds,
    )
    result = aggregate_panel(parse_panel(_panel()))
    first = result["statistical_endpoints"][0]
    assert first["point_pass"] is True
    assert first["lower_bound"] == "0.10000000000000001"
    assert first["lower_bound_strict_pass"] is False
    assert result["all_gates_pass"] is False
    assert result["decision"] == evaluator.CLOSE_WITHOUT_CONFIRMATION


def test_immutable_development_input_and_no_replace_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "_compute_simultaneous_lcbs",
        _fast_passing_lcbs,
    )
    panel_path = tmp_path / "development.panel.json"
    panel_sha = _write_immutable(panel_path, _panel())
    result_path = tmp_path / "development.result.json"
    receipt = evaluate_immutable_panel(
        panel_path=panel_path,
        expected_panel_sha256=panel_sha,
        result_path=result_path,
        split="development",
    )
    payload = result_path.read_bytes()
    result = json.loads(payload)
    assert receipt.sha256 == hashlib.sha256(payload).hexdigest()
    assert result["decision"] == OPEN_CONFIRMATION
    assert result["weight_updates"] == 0
    assert result["selection"]["checkpoint_update"] == 6000
    assert result["selection"]["checkpoint_selection"] == "update_6000_only"
    assert b"semantic_core_id" not in payload
    assert b"row_level" not in payload
    assert b"logit" not in payload
    assert result_path.stat().st_mode & 0o222 == 0
    with pytest.raises(EvaluationError, match="no-replace"):
        evaluate_immutable_panel(
            panel_path=panel_path,
            expected_panel_sha256=panel_sha,
            result_path=result_path,
            split="development",
        )


def test_writable_or_noncanonical_input_is_rejected(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.json"
    payload = canonical_json_bytes(_panel())
    panel_path.write_bytes(payload)
    panel_sha = hashlib.sha256(payload).hexdigest()
    with pytest.raises(EvaluationError, match="immutable"):
        evaluate_immutable_panel(
            panel_path=panel_path,
            expected_panel_sha256=panel_sha,
            result_path=tmp_path / "result.json",
            split="development",
        )

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text('{"schema": "wrong"}\n', encoding="ascii")
    noncanonical.chmod(0o444)
    digest = hashlib.sha256(noncanonical.read_bytes()).hexdigest()
    with pytest.raises(EvaluationError, match="canonical"):
        evaluate_immutable_panel(
            panel_path=noncanonical,
            expected_panel_sha256=digest,
            result_path=tmp_path / "result-2.json",
            split="development",
        )


def test_symlinked_input_is_rejected_before_resolution(tmp_path: Path) -> None:
    target = tmp_path / "real.panel.json"
    panel_sha = _write_immutable(target, _panel())
    link = tmp_path / "linked.panel.json"
    link.symlink_to(target)
    with pytest.raises(EvaluationError, match="immutable"):
        evaluate_immutable_panel(
            panel_path=link,
            expected_panel_sha256=panel_sha,
            result_path=tmp_path / "result.json",
            split="development",
        )


def test_confirmation_is_consumed_once_before_panel_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "_compute_simultaneous_lcbs",
        _fast_passing_lcbs,
    )
    panel_path = tmp_path / "confirmation.panel.json"
    panel_sha = _write_immutable(panel_path, _panel("confirmation"))
    evaluator_root = _digest("frozen-evaluator-runtime-root")
    authorization_path = tmp_path / "authorization.json"
    authorization_sha = _write_immutable(
        authorization_path,
        _authorization(panel_sha, evaluator_root),
    )
    claim_path = tmp_path / "opening.claim.json"
    result_path = tmp_path / "confirmation.result.json"
    receipt = evaluate_immutable_panel(
        panel_path=panel_path,
        expected_panel_sha256=panel_sha,
        result_path=result_path,
        split="confirmation",
        authorization_path=authorization_path,
        expected_authorization_sha256=authorization_sha,
        expected_evaluator_root_sha256=evaluator_root,
        opening_claim_path=claim_path,
    )
    assert receipt.byte_count > 0
    result = json.loads(result_path.read_bytes())
    assert result["decision"] == POSITIVE_DECISION
    assert result["confirmation_access_count"] == 1
    assert claim_path.exists()
    assert claim_path.stat().st_mode & 0o222 == 0

    with pytest.raises(EvaluationError, match="no-replace"):
        evaluate_immutable_panel(
            panel_path=panel_path,
            expected_panel_sha256=panel_sha,
            result_path=tmp_path / "second.result.json",
            split="confirmation",
            authorization_path=authorization_path,
            expected_authorization_sha256=authorization_sha,
            expected_evaluator_root_sha256=evaluator_root,
            opening_claim_path=claim_path,
        )


def test_failed_confirmation_attempt_still_consumes_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "_compute_simultaneous_lcbs",
        _fast_passing_lcbs,
    )
    malformed = _panel("confirmation")
    malformed["runs"][0]["checkpoint_update"] = 1000
    panel_path = tmp_path / "bad.panel.json"
    panel_sha = _write_immutable(panel_path, malformed)
    evaluator_root = _digest("frozen-evaluator-runtime-root")
    authorization_path = tmp_path / "authorization.json"
    authorization_sha = _write_immutable(
        authorization_path,
        _authorization(panel_sha, evaluator_root),
    )
    claim_path = tmp_path / "opening.claim.json"
    kwargs = {
        "panel_path": panel_path,
        "expected_panel_sha256": panel_sha,
        "result_path": tmp_path / "bad.result.json",
        "split": "confirmation",
        "authorization_path": authorization_path,
        "expected_authorization_sha256": authorization_sha,
        "expected_evaluator_root_sha256": evaluator_root,
        "opening_claim_path": claim_path,
    }
    with pytest.raises(EvaluationError, match="update-6000-only"):
        evaluate_immutable_panel(**kwargs)
    assert claim_path.exists()
    with pytest.raises(EvaluationError, match="no-replace"):
        evaluate_immutable_panel(**kwargs)


def test_confirmation_policy_and_root_mismatches_fail_closed(
    tmp_path: Path,
) -> None:
    panel_path = tmp_path / "confirmation.panel.json"
    panel_sha = _write_immutable(panel_path, _panel("confirmation"))
    evaluator_root = _digest("frozen-evaluator-runtime-root")
    authorization = _authorization(panel_sha, evaluator_root)
    authorization["retry_allowed"] = True
    authorization_path = tmp_path / "authorization.json"
    authorization_sha = _write_immutable(authorization_path, authorization)
    with pytest.raises(EvaluationError, match="policy"):
        evaluate_immutable_panel(
            panel_path=panel_path,
            expected_panel_sha256=panel_sha,
            result_path=tmp_path / "result.json",
            split="confirmation",
            authorization_path=authorization_path,
            expected_authorization_sha256=authorization_sha,
            expected_evaluator_root_sha256=evaluator_root,
            opening_claim_path=tmp_path / "claim.json",
        )
    assert not (tmp_path / "claim.json").exists()


def test_expired_confirmation_authorization_does_not_consume_claim(
    tmp_path: Path,
) -> None:
    panel_path = tmp_path / "confirmation.panel.json"
    panel_sha = _write_immutable(panel_path, _panel("confirmation"))
    evaluator_root = _digest("frozen-evaluator-runtime-root")
    authorization = _authorization(panel_sha, evaluator_root)
    authorization["expires_at_utc"] = "2000-01-01T00:00:00Z"
    authorization_path = tmp_path / "authorization.json"
    authorization_sha = _write_immutable(authorization_path, authorization)
    claim_path = tmp_path / "claim.json"
    with pytest.raises(EvaluationError, match="expired"):
        evaluate_immutable_panel(
            panel_path=panel_path,
            expected_panel_sha256=panel_sha,
            result_path=tmp_path / "result.json",
            split="confirmation",
            authorization_path=authorization_path,
            expected_authorization_sha256=authorization_sha,
            expected_evaluator_root_sha256=evaluator_root,
            opening_claim_path=claim_path,
        )
    assert not claim_path.exists()


def test_module_is_cpu_only_and_contains_no_fitting_surface() -> None:
    source = Path(evaluator.__file__).read_text(encoding="ascii")
    forbidden = (
        "import torch",
        ".backward(",
        "optimizer.step(",
        "subprocess",
        "sbatch",
        "cuda",
    )
    assert all(token not in source for token in forbidden)
