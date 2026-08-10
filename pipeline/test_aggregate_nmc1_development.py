import argparse
import json

from aggregate_nmc1_development import run


def _report(arm: str, control: str, counts: dict[str, int]) -> dict[str, object]:
    return {
        "schema": "shohin-nmc1-development-evaluation-v1",
        "arm": arm,
        "control": control,
        "holdout_used": False,
        "public_test_opened": False,
        "counts": {"rows": 666, **counts},
        "exhausted": 0,
        "details": [{"identity_sha256": f"{index:064x}"} for index in range(666)],
    }


def test_frozen_gate_uses_normal_correct_multi_digit_intersection(tmp_path) -> None:
    reports = {
        "program": _report(
            "program",
            "normal",
            {
                "syntax_valid": 666,
                "normal:valid": 666,
                "normal:correct": 500,
                "program_exact": 450,
                "normal_correct_multi_digit_rows": 400,
                "carry_reset:normal_correct_multi_digit_correct": 200,
                "opcode_permuted:correct": 10,
            },
        ),
        "shuffled": _report("program", "source_shuffled", {"normal:correct": 20}),
        "direct": _report("direct", "normal", {"answer_correct": 450}),
    }
    paths = {}
    for name, report in reports.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report))
        paths[name] = path
    output = tmp_path / "aggregate.json"
    result = run(
        argparse.Namespace(
            program=paths["program"],
            shuffled=paths["shuffled"],
            direct=paths["direct"],
            output=output,
        )
    )
    assert result["metrics"]["carry_reset_multi_digit_rate"] == 0.5
    assert result["metrics"]["normal_multi_digit_rate"] == 1.0
    assert result["overall_pass"] is True
    assert output.exists()
