import json

from compare_dset1_stage0 import run


class Args:
    pass


def _arm(name, source_ids, correct, script_accuracy=1.0):
    rows = len(source_ids)
    half = rows // 2
    return {
        "schema": "shohin-dset1-span-edit-evaluation-merged-v1",
        "status": "complete",
        "arm": name,
        "holdout_used": False,
        "row_count": rows,
        "pair_count": half,
        "data_sha256": "data",
        "data_report_sha256": "report",
        "script_exact": int(rows * script_accuracy),
        "execution_correct": correct,
        "counterfactual_consistency": script_accuracy,
        "family_metrics": {
            "numeric_final": {"script_exact_accuracy": script_accuracy}
        },
        "member_metrics": {
            "clean": {"execution_correct_accuracy": 1.0},
            "fault": {"execution_correct_accuracy": max(0.0, (correct - half) / half)},
        },
        "execution_errors": {},
        "max_token_exhausted": 0,
        "results": [
            {"source_dseo1_identity_sha256": identity} for identity in source_ids
        ],
    }


def test_compare_passes_conjunctive_fixture(tmp_path) -> None:
    ids = [f"i{i}" for i in range(200)]
    paths = {}
    for name, correct, scripts in (
        ("aligned", 195, 0.95),
        ("swapped", 100, 0.50),
        ("hidden", 100, 0.50),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_arm(name, ids, correct, scripts)))
        paths[name] = path
    final = tmp_path / "final.json"
    final.write_text(
        json.dumps(
            {
                "schema": "shohin-dseo1-paired-evaluation-merged-v1",
                "status": "complete",
                "arm": "final_only",
                "results": [
                    {"identity_sha256": identity, "answer_correct": index < 180}
                    for index, identity in enumerate(ids)
                ],
            }
        )
    )
    args = Args()
    args.aligned, args.swapped, args.hidden = paths["aligned"], paths["swapped"], paths["hidden"]
    args.final_only = final
    args.output = tmp_path / "comparison.json"
    report = run(args)
    assert report["passed"]
    assert report["margins"]["aligned_minus_final_only"] == 15
