from rescore_verified_code_candidates import rescore


def test_rescore_reexecutes_code_against_bound_bank() -> None:
    row = {
        "task": "mbpp",
        "identity_sha256": "id",
        "text": "Add two integers.",
        "test_list": ["assert add(2, 3) == 5"],
        "test_setup_code": "",
    }
    candidates = [
        {
            "task": "mbpp",
            "identity_sha256": "id",
            "sample_index": 0,
            "completion": "def add(a, b):\n    return a + b",
            "correct": False,
        }
    ]
    rescored, report = rescore(candidates, [row], 1)
    assert rescored[0]["correct"]
    assert report["label_transitions"] == {"0->1": 1}
