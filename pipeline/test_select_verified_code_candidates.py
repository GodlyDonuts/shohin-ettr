from select_verified_code_candidates import select, visible_humaneval_result


def _candidate(index: int, correct: bool, completion: str, passed: bool = False):
    return {
        "identity_sha256": "id",
        "task": "humaneval",
        "sample_index": index,
        "correct": correct,
        "completion": completion,
        "generated_tokens": 20 + index,
        "execution": {"passed": passed},
    }


def test_visible_humaneval_result_uses_prompt_examples() -> None:
    row = {
        "task": "humaneval",
        "prompt": 'def add(a, b):\n    """\n    >>> add(2, 3)\n    5\n    """\n',
        "entry_point": "add",
    }
    good = visible_humaneval_result(row, "def add(a, b):\n    return a + b", 1)
    bad = visible_humaneval_result(row, "def add(a, b):\n    return a - b", 1)
    assert good["visible_tests_passed"]
    assert not bad["visible_tests_passed"]


def test_selector_chooses_visible_pass_without_reading_correctness() -> None:
    row = {
        "task": "humaneval",
        "identity_sha256": "id",
        "prompt": 'def add(a, b):\n    """\n    >>> add(2, 3)\n    5\n    """\n',
        "entry_point": "add",
    }
    candidates = [
        _candidate(0, False, "def add(a, b):\n    return a - b"),
        _candidate(1, True, "def add(a, b):\n    return a + b"),
    ]
    report = select(candidates, [row], 1)
    assert report["selected_correct"] == 1
    assert report["results"][0]["selected_sample_index"] == 1
    assert not report["selector_reads_hidden_tests"]


def test_mbpp_selector_uses_visible_execution_result() -> None:
    row = {
        "task": "mbpp",
        "identity_sha256": "id",
        "text": "Add two integers.",
        "test_list": ["assert add(2, 3) == 5"],
    }
    candidates = [
        {**_candidate(0, False, "def add(a, b): return a - b"), "task": "mbpp"},
        {
            **_candidate(1, True, "def add(a, b): return a + b", passed=True),
            "task": "mbpp",
        },
    ]
    report = select(candidates, [row], 1)
    assert report["selected_correct"] == 1
    assert report["results"][0]["selection"] == "visible_task_tests"
