from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from source_deleted_multifamily_machine_board import (  # noqa: E402
    compile_source,
    generate_episode,
)
from run_multifamily_renderer_curriculum_smoke import (  # noqa: E402
    _target_first_augmentation,
    run_renderer_curriculum_smoke,
)


def test_target_first_augmentation_preserves_law_and_answer() -> None:
    row = generate_episode(
        seed=88,
        split="train",
        family="affine_modular",
        renderer=1,
        cell="fit",
    )
    augmented = _target_first_augmentation(row)
    original = compile_source(row.candidate.source)
    assert augmented.supervisor.law_sha256 == row.supervisor.law_sha256
    assert augmented.supervisor.answer == row.supervisor.answer
    assert len(compile_source(row.candidate.source).transition) == len(
        original.transition
    )
    assert "<-{" in augmented.candidate.source
    assert "<=" not in augmented.candidate.source
    assert "@" not in augmented.candidate.query


def test_tiny_renderer_curriculum_smoke_is_equal_budget() -> None:
    report = run_renderer_curriculum_smoke(
        seed=123,
        steps=2,
        width=32,
        layers=1,
        learning_rate=1e-3,
        device=torch.device("cpu"),
    )
    assert report["candidate_time_oracle_calls"] == 0
    assert report["candidate_time_search_calls"] == 0
    assert report["candidate_time_verifier_calls"] == 0
    assert report["equal_budget"]["initialization_identical"]
    assert report["equal_budget"]["parameters_per_arm"] > 0
    assert report["development"]["renderer_curriculum_treatment"]["total"] == 24


def test_tiny_renderer_curriculum_family_holdout() -> None:
    report = run_renderer_curriculum_smoke(
        seed=321,
        steps=1,
        width=32,
        layers=1,
        learning_rate=1e-3,
        device=torch.device("cpu"),
        held_out_family="permutation",
    )
    assert report["held_out_family"] == "permutation"
    assert report["equal_budget"]["base_rows"] == 24
    assert report["development"]["renderer_curriculum_treatment"]["total"] == 8
