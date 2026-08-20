from eval_dense_public_evalplus import extract_python_solution


def test_extract_python_solution_prefers_last_fenced_program() -> None:
    text = "Draft:\n```python\nprint('old')\n```\nFinal:\n```python\nprint('new')\n```"
    assert extract_python_solution(text) == "print('new')"


def test_extract_python_solution_keeps_plain_program() -> None:
    assert extract_python_solution("def f():\n    return 1\n") == "def f():\n    return 1"
