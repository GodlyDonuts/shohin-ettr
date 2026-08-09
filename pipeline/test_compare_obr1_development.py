from compare_obr1_development import compare


def fixtures(score=300, math=75, logic=215, code=10):
    report = {
        "schema": "shohin-idr1-revision-evaluation-v1",
        "status": "complete",
        "metrics": {
            "overall": {"generated_correct": score},
            "math500": {"generated_correct": math},
            "bbh_logic": {"generated_correct": logic},
            "mbpp": {"generated_correct": code},
        },
    }
    fit = {
        "schema": "shohin-rme1-product-training-v1",
        "status": "complete",
        "architecture": "shohin-rme1-moe-revision-v1",
        "rme1_draft_control": "draft_unavailable",
        "rme1_config": {"mode": "shared", "controlled_layers": 16, "rank": 18},
        "trainable_parameters": 1_179_648,
        "updates": 2048,
        "protected_router_expert_trainables": 0,
        "sequence_custody": {"overflow_rows": 0},
    }
    data = {
        "schema": "shohin-obr1-broad-owner-data-report-v1",
        "status": "complete",
        "holdout_used": False,
        "zero_exact_development_overlap": True,
        "zero_ngram_development_overlap": True,
        "complete_retention": True,
    }
    return report, fit, data


def test_gate_passes_only_at_all_frozen_floors():
    report, fit, data = fixtures()
    assert compare(report, fit, data)["status"] == "pass"
    report, fit, data = fixtures(code=9)
    assert compare(report, fit, data)["status"] == "fail"
