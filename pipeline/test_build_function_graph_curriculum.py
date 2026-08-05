from __future__ import annotations

import pytest

from build_function_graph_curriculum import (
    FunctionGraphCurriculumError,
    build_curriculum,
)


def _generated(index: int) -> dict:
    return {
        "question": f"generated {index}",
        "response": f"return {index}",
        "split": "train",
        "family": "list" if index % 2 else "string",
    }


def _anchor(index: int) -> dict:
    return {"question": f"anchor {index}", "response": f"return {index}"}


def test_curriculum_is_deterministic_and_anchored() -> None:
    first, report = build_curriculum(
        [_generated(index) for index in range(4)],
        [_anchor(index) for index in range(2)],
        anchor_repeats=3,
        seed=31,
    )
    second, _ = build_curriculum(
        [_generated(index) for index in range(4)],
        [_anchor(index) for index in range(2)],
        anchor_repeats=3,
        seed=31,
    )
    assert first == second
    assert report["rows"] == 10
    assert report["anchor_materialized_rows"] == 6
    assert (
        sum(row["curriculum_origin"] == "generated_function_graph" for row in first)
        == 4
    )


def test_curriculum_rejects_cross_source_overlap() -> None:
    with pytest.raises(FunctionGraphCurriculumError, match="overlap"):
        build_curriculum(
            [_generated(0)],
            [{"question": "generated 0", "response": "return 0"}],
            anchor_repeats=1,
            seed=31,
        )
