import argparse
import json

import compare_ctsr1_development as module


def write(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def evaluation(correct, math, logic, code, routed=False):
    report = {
        "schema": "shohin-idr1-revision-evaluation-v1", "status": "complete",
        "split": "development", "full_row_count": 1289,
        "merged_from_shards": True, "shard_count": 8,
        "ecr_code_intervention": "normal", "model_revision": "r",
        "data_sha256": "d", "data_report_sha256": "dr",
        "metrics": {
            "overall": {"generated_correct": correct, "total": 1289},
            "math500": {"generated_correct": math, "total": 623},
            "bbh_logic": {"generated_correct": logic, "total": 637},
            "mbpp": {"generated_correct": code, "total": 29},
        },
    }
    if routed:
        layer = {
            "tokens": 100, "load_entropy": 0.95,
            "mean_token_entropy_normalized": 0.8,
            "route_probability_l1_mean": 0.1, "top1_change_rate": 0.02,
            "active_experts": 64, "mean_state_norm": 2.0,
            "mean_residual_norm": 1.0,
        }
        report["routing_receipts"] = [
            {"layers": [dict(layer) for _ in range(16)]} for _ in range(8)
        ]
    return report


def fit(mode):
    return {
        "schema": "shohin-ctsr1-product-training-v1", "status": "complete",
        "updates": 256, "selected_rows": 9651, "charged_tokens": 338620,
        "trainable_parameters": 1594752,
        "adapter_macs_per_token_per_layer": 488576,
        "protected_router_expert_trainables": 0,
        "ctsr1_draft_control": "normal",
        "ctsr1_config": {
            "mode": mode, "controlled_layers": 16, "state_width": 64,
            "head_width": 32, "residual_rank": 18, "residual_alpha": 18.0,
            "router_scale": 1.0, "entropy_floor": 0.8,
            "collapse_weight": 0.01,
        },
        "sequence_custody": {
            "overflow_rows": 0, "source_retention": 1.0,
            "draft_retention": 1.0, "target_retention": 1.0,
        },
    }


def args(tmp_path, monkeypatch):
    paths = {name: tmp_path / f"{name}.json" for name in (
        "treatment_report", "temporal_shared_report", "static_shared_report",
        "treatment_fit", "temporal_shared_fit", "output",
    )}
    write(paths["treatment_report"], evaluation(300, 70, 220, 10, routed=True))
    write(paths["temporal_shared_report"], evaluation(270, 65, 197, 8))
    write(paths["static_shared_report"], evaluation(248, 60, 181, 7))
    write(paths["treatment_fit"], fit("temporal_router"))
    write(paths["temporal_shared_fit"], fit("temporal_shared"))
    static_hash = module.sha256_file(paths["static_shared_report"])
    monkeypatch.setattr(module, "STATIC_SHARED_SHA256", static_hash)
    return argparse.Namespace(**paths)


def test_pass(tmp_path, monkeypatch):
    result = module.compare(args(tmp_path, monkeypatch))
    assert result["causal_controls_authorized"] is True


def test_kill_on_weak_route_change(tmp_path, monkeypatch):
    value = args(tmp_path, monkeypatch)
    treatment = json.loads(value.treatment_report.read_text())
    for receipt in treatment["routing_receipts"]:
        for layer in receipt["layers"]:
            layer["top1_change_rate"] = 0.001
    write(value.treatment_report, treatment)
    result = module.compare(value)
    assert result["causal_controls_authorized"] is False

