from build_verified_code_repair_curriculum import (
    build_curriculum,
    mutation_candidates,
)


def _raw() -> dict:
    return {
        "task_id": 1,
        "text": "Increment an integer.",
        "code": "def inc(x):\n    return x + 1",
        "test_setup_code": "",
        "test_list": ["assert inc(2) == 3", "assert inc(-1) == 0"],
    }


def _anchor() -> dict:
    return {
        "question": "Increment an integer.",
        "response": "def inc(x):\n    return x + 1",
        "source": "mbpp_train",
    }


def test_mutations_change_executable_program() -> None:
    candidates = mutation_candidates(_raw()["code"], "seed")
    assert candidates
    assert len({program for program, _ in candidates}) == len(candidates)
    assert all(program != _raw()["code"] for program, _ in candidates)
    assert any("return x - 1" in program for program, _ in candidates)
    assert any("return x + 2" in program for program, _ in candidates)


def test_builds_executed_bug_to_fix_trajectory() -> None:
    rows, report = build_curriculum(
        [_raw()],
        [_anchor()],
        mutations_per_source=2,
        max_candidates=8,
        timeout_seconds=2,
        seed=7,
        workers=1,
    )
    assert rows
    assert report["sources_with_repairs"] == 1
    assert report["repair_rows"] == len(rows)
    assert all(not row["failure_execution"]["passed"] for row in rows)
    assert all(row["response"] == _raw()["code"] for row in rows)
    assert all("Tests:\nassert inc(2) == 3" in row["question"] for row in rows)
