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


def test_curriculum_caps_semantic_repetition_and_family_size() -> None:
    rows = [_generated(index) for index in range(8)]
    for index, row in enumerate(rows):
        row["graph"] = {"operation": (index // 2) % 2}
        row["global_identity"] = index
        row["verification_sha256"] = str(index)
    selected, report = build_curriculum(
        rows,
        [_anchor(0)],
        anchor_repeats=1,
        seed=31,
        generated_max_per_graph=1,
        generated_max_per_family=2,
    )
    assert report["generated_selected_rows"] == 4
    assert report["generated_unique_graphs_selected"] == 4
    assert report["family_counts"] == {"list": 2, "string": 2}
    assert len(selected) == 5
