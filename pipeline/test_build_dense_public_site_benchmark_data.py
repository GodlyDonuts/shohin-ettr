from build_dense_public_site_benchmark_data import identity, lcb_prompt


def test_livecodebench_prompt_uses_official_generic_shape() -> None:
    prompt = lcb_prompt(
        {"question_content": "Solve it", "starter_code": "def solve():\n    pass"}
    )
    assert prompt.startswith("### Question:\nSolve it")
    assert "```python\ndef solve():" in prompt
    assert prompt.endswith("### Answer: (use the provided format with backticks)\n")


def test_identity_is_stable_to_whitespace_only_prompt_changes() -> None:
    assert identity("bench", "1", "a  b\n c") == identity("bench", "1", "a b c")
