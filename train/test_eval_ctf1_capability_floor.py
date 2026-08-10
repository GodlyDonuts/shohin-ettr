from pathlib import Path


def test_unadapted_owner_uses_raw_generation_path() -> None:
    source = (Path(__file__).parent / "eval_ctf1_capability_floor.py").read_text()
    call = source.split("completions, usage = _generate_completions(", 1)[1].split(
        ")", 1
    )[0]
    assert "\n            False,\n" in call
