import argparse
import json

from compare_rme1_development import compare


def _write(path, payload):
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _routes():
    layer = {"active_revision_experts": 4, "load_entropy": 0.95}
    return [{"layers": [dict(layer) for _ in range(16)]} for _ in range(8)]


def _evaluation(correct, math, logic, code, routed=False):
    value = {
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
    if routed:
        value["routing_receipts"] = _routes()
    return value


def _fit(mode, rank, parameters):
    return {
        "schema": "shohin-rme1-product-training-v1",
        "status": "complete",
        "updates": 256,
        "selected_rows": 9651,
        "trainable_parameters": parameters,
        "rme1_config": {
            "mode": mode,
            "controlled_layers": 16,
            "rank": rank,
            "alpha": float(rank),
            "revision_experts": 4,
            "revision_top_k": 2,
            "balance_weight": 0.01,
        },
        "rme1_draft_control": "normal",
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
    _write(paths["treatment_report"], _evaluation(310, 75, 225, 10, routed=True))
    _write(paths["shared_flop_report"], _evaluation(270, 68, 194, 8))
    _write(paths["shared_parameter_report"], _evaluation(280, 70, 202, 8))
    _write(paths["treatment_fit"], _fit("routed", 8, 2228224))
    _write(paths["shared_flop_fit"], _fit("shared", 18, 1179648))
    _write(paths["shared_parameter_fit"], _fit("shared", 34, 2228224))
    return argparse.Namespace(**paths)


def test_pass(tmp_path):
    result = compare(_args(tmp_path))
    assert result["stage_two_authorized"] is True
    assert result["minimum_load_entropy"] == 0.95


def test_kill_on_route_collapse(tmp_path):
    args = _args(tmp_path)
    treatment = json.loads(args.treatment_report.read_text())
    treatment["routing_receipts"][0]["layers"][0]["active_revision_experts"] = 3
    _write(args.treatment_report, treatment)
    result = compare(args)
    assert result["stage_two_authorized"] is False
    assert result["close_rme1_if_false"] is True


def test_kill_on_nonfinite_route_entropy(tmp_path):
    args = _args(tmp_path)
    treatment = json.loads(args.treatment_report.read_text())
    treatment["routing_receipts"][0]["layers"][0]["load_entropy"] = float("nan")
    _write(args.treatment_report, treatment)
    result = compare(args)
    assert result["stage_two_authorized"] is False


def test_reject_wrong_token_budget(tmp_path):
    args = _args(tmp_path)
    treatment = json.loads(args.treatment_fit.read_text())
    treatment["charged_tokens"] -= 1
    _write(args.treatment_fit, treatment)
    try:
        compare(args)
    except Exception as error:
        assert "fit receipt differs" in str(error)
    else:
        raise AssertionError("comparison accepted a non-frozen token budget")
