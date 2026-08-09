import argparse
import json

from compare_ser1_development import compare


def _write(path, payload):
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _evaluation(correct, math, logic, code):
    return {
        "schema": "shohin-idr1-revision-evaluation-v1",
        "status": "complete",
        "split": "development",
        "full_row_count": 1289,
        "merged_from_shards": True,
        "shard_count": 8,
        "ecr_code_intervention": "normal",
        "model_revision": "revision",
        "data_sha256": "data",
        "data_report_sha256": "report",
        "metrics": {
            "overall": {"generated_correct": correct, "total": 1289},
            "math500": {"generated_correct": math, "total": 623},
            "bbh_logic": {"generated_correct": logic, "total": 637},
            "mbpp": {"generated_correct": code, "total": 29},
        },
    }


def _fit(mode, rank, parameters):
    return {
        "schema": "shohin-ser1-product-training-v1",
        "status": "complete",
        "updates": 256,
        "selected_rows": 9651,
        "trainable_parameters": parameters,
        "ser1_config": {
            "mode": mode,
            "controlled_layers": 16,
            "rank": rank,
            "alpha": float(rank),
        },
        "ser1_draft_control": "normal",
        "protected_router_expert_trainables": 0,
        "sequence_custody": {
            "overflow_rows": 0,
            "source_retention": 1.0,
            "draft_retention": 1.0,
            "target_retention": 1.0,
        },
        "charged_tokens": 338620,
    }


def _args(root):
    names = (
        "treatment_report",
        "shared_flop_report",
        "shared_parameter_report",
        "treatment_fit",
        "shared_flop_fit",
        "shared_parameter_fit",
        "output",
    )
    paths = {name: root / f"{name}.json" for name in names}
    _write(paths["treatment_report"], _evaluation(300, 70, 220, 10))
    _write(paths["shared_flop_report"], _evaluation(260, 65, 188, 7))
    _write(paths["shared_parameter_report"], _evaluation(270, 68, 194, 8))
    _write(paths["treatment_fit"], _fit("selected_expert", 1, 4194304))
    _write(paths["shared_flop_fit"], _fit("shared", 8, 524288))
    _write(paths["shared_parameter_fit"], _fit("shared", 64, 4194304))
    return argparse.Namespace(**paths)


def test_pass(tmp_path):
    result = compare(_args(tmp_path))
    assert result["stage_two_authorized"] is True
    assert result["holdout_authorized"] is False


def test_kill_on_matched_control(tmp_path):
    args = _args(tmp_path)
    _write(args.shared_parameter_report, _evaluation(279, 68, 203, 8))
    result = compare(args)
    assert result["stage_two_authorized"] is False
    assert result["close_ser1_if_false"] is True
