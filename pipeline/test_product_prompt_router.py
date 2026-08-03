from pipeline.product_prompt_router import route
from pipeline.product_prompt_router import route_reports
from pipeline.product_prompt_router import calibrate_threshold
from pipeline.product_prompt_router import train_router
from pipeline.product_prompt_router import training_label


def test_training_policy_splits_arithmetic_from_string_procedures() -> None:
    assert training_label({"training_group": "math", "question": "q"}) == "dense_residual"
    assert training_label({"training_group": "code", "question": "q"}) == "baseline"
    assert training_label(
        {"training_group": "procedural", "question": "Calculate 3 + 4"}
    ) == "dense_residual"
    assert training_label(
        {"training_group": "procedural", "question": "Sort these words"}
    ) == "baseline"
    assert training_label({"training_group": "teacher", "question": "q"}) is None


def test_router_learns_distinct_prompt_domains() -> None:
    rows = []
    for index in range(20):
        rows.append((f"prove algebra theorem equation {index}", "dense_residual"))
        rows.append((f"write python function code {index}", "baseline"))
    model, report = train_router(rows, min_feature_count=1, max_features=1000)

    assert report["validation"]["accuracy"] == 1.0
    assert route(model, "solve algebra equation")[0] == "dense_residual"
    assert route(model, "implement python function")[0] == "baseline"


def test_route_reports_selects_the_predicted_arm() -> None:
    model = {
        "decision_threshold": 0.0,
        "feature_weights": {"u:math": 2.0, "u:python": -2.0},
        "schema": "shohin-product-prompt-router-v1",
    }
    baseline = {
        "results": [
            {"identity_sha256": "a", "question": "math problem", "gold": "1", "correct": False},
            {"identity_sha256": "b", "question": "python problem", "gold": "2", "correct": True},
        ],
        "task": "test",
    }
    dense = {
        "results": [
            {"identity_sha256": "a", "question": "math problem", "gold": "1", "correct": True},
            {"identity_sha256": "b", "question": "python problem", "gold": "2", "correct": False},
        ],
        "task": "test",
    }

    routed = route_reports(model, baseline, dense)

    assert routed["correct"] == 2
    assert routed["route_counts"] == {"baseline": 1, "dense_residual": 1}


def test_calibration_selects_dense_without_regressing_other_domains() -> None:
    model = {
        "decision_threshold": 0.0,
        "feature_weights": {"u:dense": 2.0, "u:base": -2.0},
        "schema": "shohin-product-prompt-router-v1",
    }
    reports = {}
    for task in ("gsm8k", "math500", "humaneval", "mbpp", "gpqa", "bbh_logic"):
        baseline_results = []
        dense_results = []
        for index in range(10):
            dense_prompt = task in {"gsm8k", "math500", "gpqa"}
            question = ("dense" if dense_prompt else "base") + f" {task} {index}"
            common = {"identity_sha256": f"{task}-{index}", "question": question}
            baseline_results.append({**common, "correct": not dense_prompt})
            dense_results.append({**common, "correct": dense_prompt})
        reports[task] = (
            {"results": baseline_results},
            {"results": dense_results},
        )

    calibrated, report = calibrate_threshold(model, reports)

    assert calibrated["decision_threshold"] > -2.0
    assert report["selected"]["macro_accuracy"] == 1.0
