from hf_dpr1_panel_ceiling import normalize


def test_normalize_collapses_only_case_and_whitespace():
    assert normalize("  Answer:  42\n") == "answer: 42"
    assert normalize("42") != normalize("41")

