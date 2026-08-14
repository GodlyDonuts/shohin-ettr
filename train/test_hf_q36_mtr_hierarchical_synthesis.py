from __future__ import annotations

import json
from pathlib import Path

import pytest

import hf_q36_mtr_hierarchical_synthesis as module


def _row(identity: str, schema: str = module.SCHEMA) -> dict:
    return {
        "schema": schema,
        "identity_sha256": identity,
        "split": "development",
        "task": "math500",
        "completion": "A nonempty completion",
        "generated_tokens": 4,
        "max_token_exhausted": False,
    }


def test_hierarchical_prompt_preserves_primary_and_hides_architecture_names() -> None:
    prompt = module.hierarchical_prompt(
        "Original question", "integrated answer", "stacked answer", "review answer"
    )
    assert "Preserve Candidate A unless" in prompt
    assert "integrated answer" in prompt
    assert "stacked answer" in prompt
    assert "review answer" in prompt
    assert "owner_71" not in prompt
    assert "development" not in prompt


def test_hierarchical_prompt_rejects_empty_input() -> None:
    with pytest.raises(module.Q36MTRHierarchicalSynthesisError):
        module.hierarchical_prompt("question", "", "other", "review")


def test_incumbent_challenger_prompt_is_asymmetric_and_task_agnostic() -> None:
    prompt = module.incumbent_challenger_prompt(
        "Original question", "incumbent", "deeper synthesis", "direct synthesis"
    )
    assert "should be preserved unless a concrete" in prompt
    assert "never change A merely because alternatives agree" in prompt
    assert "recompute the disputed reasoning" in prompt
    assert "math500" not in prompt
    assert "development" not in prompt


def test_incumbent_cyclic_prompt_preserves_incumbent_without_voting() -> None:
    prompt = module.incumbent_cyclic_prompt(
        "Original question", "incumbent", "cyclic one", "cyclic two"
    )
    assert "should be preserved unless a concrete" in prompt
    assert "never change A because of surface agreement or voting" in prompt
    assert "cyclic one" in prompt and "cyclic two" in prompt


def test_incumbent_interpolation_prompt_is_conservative_and_task_agnostic() -> None:
    prompt = module.incumbent_interpolation_prompt(
        "Original question", "incumbent", "interpolated synthesis", "direct synthesis"
    )
    assert "should be preserved unless a concrete" in prompt
    assert "do not vote" in prompt
    assert "interpolated synthesis" in prompt
    assert "math500" not in prompt
    assert "development" not in prompt


def _adjudication_rows(task: str, completions: tuple[str, ...]) -> dict[str, dict]:
    return {
        arm: {
            "identity_sha256": "0" * 64,
            "task": task,
            "completion": completion,
        }
        for arm, completion in zip(module.ADJUDICATION_ARMS, completions, strict=True)
    }


def test_multi_trajectory_adjudication_preserves_unanimous_answer() -> None:
    rows = _adjudication_rows("math500", (r"\boxed{7}",) * 6)
    selected, prompt, plan = module.multi_trajectory_adjudication_plan(
        "Original question", rows
    )
    assert selected == "hierarchy"
    assert prompt is None
    assert plan == {
        "decision": "preserve_unanimous_answer",
        "unique_answers": 1,
        "maximum_support": 6,
    }


def test_multi_trajectory_adjudication_prompts_only_on_disagreement() -> None:
    rows = _adjudication_rows("bbh_logic", ("A", "B", "B", "C", "B", "A"))
    selected, prompt, plan = module.multi_trajectory_adjudication_plan(
        "Original question", rows
    )
    assert selected is None
    assert prompt is not None
    assert "supported by 3 of 6" in prompt
    assert "supported by 2 of 6" in prompt
    assert "Support counts are evidence, not proof" in prompt
    assert "bbh_logic" not in prompt
    assert plan == {
        "decision": "model_owned_disagreement_adjudication",
        "unique_answers": 3,
        "maximum_support": 3,
    }


def test_multi_trajectory_adjudication_preserves_executable_control() -> None:
    rows = _adjudication_rows("mbpp", tuple(f"code {i}" for i in range(6)))
    selected, prompt, plan = module.multi_trajectory_adjudication_plan(
        "Original question", rows
    )
    assert selected == "interpolation"
    assert prompt is None
    assert plan["decision"] == "preserve_executable_control"


def _guidance(selected: str, completion: str, labels_read: int = 0) -> dict:
    return {
        "schema": module.SCHEMA,
        "identity_sha256": "0" * 64,
        "task": "math500",
        "completion": completion,
        "nested_pattern_consensus": {
            "schema": "shohin-q36-mtr-nested-pattern-consensus-v1",
            "selected": selected,
            "estimated_reliability": -0.7,
            "heldout_identity_labels_read": labels_read,
        },
    }


def test_guided_adjudication_puts_crossfit_incumbent_first() -> None:
    rows = _adjudication_rows(
        "math500",
        (
            r"hierarchy \boxed{1}",
            r"interpolation \boxed{2}",
            r"direct \boxed{2}",
            r"offset \boxed{3}",
            r"level \boxed{2}",
            r"challenger \boxed{1}",
        ),
    )
    selected, prompt, plan = module.guided_multi_trajectory_adjudication_plan(
        "Original question", rows, _guidance("interpolation", r"\boxed{2}")
    )
    assert selected is None
    assert prompt is not None
    assert "Proposal 1 — cross-fitted incumbent" in prompt
    assert prompt.index("interpolation") < prompt.index("hierarchy")
    assert "useful prior, not proof" in prompt
    assert "trained without this identity or its shard" in prompt
    assert plan["guidance_selected"] == "interpolation"
    assert plan["guidance_heldout_identity_labels_read"] == 0


def test_guided_adjudication_rejects_heldout_label_access() -> None:
    rows = _adjudication_rows("math500", (r"\boxed{1}", r"\boxed{2}") * 3)
    with pytest.raises(module.Q36MTRHierarchicalSynthesisError):
        module.guided_multi_trajectory_adjudication_plan(
            "Original question",
            rows,
            _guidance("interpolation", r"\boxed{2}", labels_read=1),
        )


def test_mode_contract_freezes_geometry_and_seed() -> None:
    assert module.mode_contract("retention_controls")["path_counts"] == (16, 1, 8)
    challenger = module.mode_contract("incumbent_challenger")
    assert challenger["path_counts"] == (16, 16, 16)
    assert challenger["seed"] == module.INCUMBENT_CHALLENGER_SEED
    cyclic = module.mode_contract("incumbent_cyclic")
    assert cyclic["path_counts"] == (16, 16, 16)
    assert cyclic["seed"] == module.INCUMBENT_CYCLIC_SEED
    interpolation = module.mode_contract("incumbent_interpolation")
    assert interpolation["path_counts"] == (16, 16, 16)
    assert interpolation["seed"] == module.INCUMBENT_INTERPOLATION_SEED
    adjudication = module.mode_contract("multi_trajectory_adjudication")
    assert adjudication["path_counts"] == (16, 16, 16, 16, 16, 16)
    assert adjudication["seed"] == module.MULTI_TRAJECTORY_ADJUDICATION_SEED
    guided = module.mode_contract("guided_multi_trajectory_adjudication")
    assert guided["path_counts"] == (16, 16, 16, 16, 16, 16)
    assert guided["seed"] == module.GUIDED_MULTI_TRAJECTORY_ADJUDICATION_SEED
    with pytest.raises(module.Q36MTRHierarchicalSynthesisError):
        module.mode_contract("unknown")


def test_candidate_loader_accepts_model_and_control_schemas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "ROWS", 2)
    path = tmp_path / "candidates.jsonl"
    rows = [_row("0" * 64), _row("1" * 64, "shohin-q36-mtr-candidate-v1")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    loaded = module.load_candidate_group([path], expected_paths=1)
    assert list(loaded) == ["0" * 64, "1" * 64]


def test_candidate_loader_rejects_duplicate_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "ROWS", 1)
    path = tmp_path / "candidates.jsonl"
    row = _row("0" * 64)
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(module.Q36MTRHierarchicalSynthesisError):
        module.load_candidate_group([path], expected_paths=1)
