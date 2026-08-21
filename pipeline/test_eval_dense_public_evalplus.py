import pytest

from dense_public_official_scoring import OfficialScoringError
from eval_dense_public_evalplus import (
    extract_python_solution,
    normalize_evalplus_task_id,
)


def test_extract_python_solution_prefers_last_fenced_program() -> None:
    text = "Draft:\n```python\nprint('old')\n```\nFinal:\n```python\nprint('new')\n```"
    assert extract_python_solution(text) == "print('new')"


def test_extract_python_solution_keeps_plain_program() -> None:
    assert (
        extract_python_solution("def f():\n    return 1\n") == "def f():\n    return 1"
    )


@pytest.mark.parametrize(
    ("benchmark", "raw_task_id", "expected"),
    [
        ("humaneval_plus", "HumanEval/0", "HumanEval/0"),
        ("humaneval_plus", 0, "HumanEval/0"),
        ("mbpp_plus", "Mbpp/2", "Mbpp/2"),
        ("mbpp_plus", 2, "Mbpp/2"),
    ],
)
def test_normalize_evalplus_task_id_uses_official_namespace(
    benchmark: str, raw_task_id: object, expected: str
) -> None:
    assert normalize_evalplus_task_id(benchmark, raw_task_id) == expected


@pytest.mark.parametrize(
    ("benchmark", "raw_task_id"),
    [("unknown", 2), ("mbpp_plus", "Other/2"), ("mbpp_plus", "")],
)
def test_normalize_evalplus_task_id_rejects_ambiguous_identity(
    benchmark: str, raw_task_id: object
) -> None:
    with pytest.raises(OfficialScoringError):
        normalize_evalplus_task_id(benchmark, raw_task_id)
