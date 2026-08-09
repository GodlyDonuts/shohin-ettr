import json
from pathlib import Path

from compare_ecr1_development import compare


class Args:
    pass


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def evaluation(correct: int, control: str = "normal") -> dict:
    return {
        "schema": "shohin-idr1-revision-evaluation-v1",
        "status": "complete",
        "split": "development",
        "full_row_count": 1289,
        "merged_from_shards": True,
        "shard_count": 8,
        "model_revision": "pinned",
        "data_sha256": "data",
        "data_report_sha256": "report",
        "ecr_code_intervention": control,
        "metrics": {
            "overall": {"generated_correct": correct, "total": 1289},
            "math500": {"generated_correct": 80, "total": 623},
            "bbh_logic": {"generated_correct": 180, "total": 637},
            "mbpp": {"generated_correct": 8, "total": 29},
        },
    }


def fit(parameters: int, mode: str, draft_control: str) -> dict:
    return {
        "schema": "shohin-ecr1-product-training-v1",
        "status": "complete",
        "updates": 256,
        "selected_rows": 9651,
        "trainable_parameters": parameters,
        "ecr1_config": {"mode": mode},
        "ecr1_draft_control": draft_control,
        "protected_router_expert_trainables": 0,
        "sequence_custody": {"overflow_rows": 0},
        "charged_tokens": 338620,
    }


def test_conjunctive_gate_passes_only_with_causal_margins(tmp_path: Path) -> None:
    args = Args()
    arms = {
        "treatment_report": evaluation(300),
        "shared_report": evaluation(250),
        "draft_report": evaluation(247),
        "zero_report": evaluation(280, "zero"),
        "mean_report": evaluation(279, "mean"),
        "permutation_report": evaluation(275, "permutation"),
    }
    for name, report in arms.items():
        setattr(args, name, write(tmp_path / f"{name}.json", report))
    args.unchanged_report = write(tmp_path / "unchanged.json", evaluation(191))
    args.mtr_report = write(tmp_path / "mtr.json", evaluation(204))
    args.treatment_fit = write(
        tmp_path / "treatment_fit.json", fit(515840, "expert_conditioned", "normal")
    )
    args.shared_fit = write(
        tmp_path / "shared_fit.json", fit(524288, "shared", "normal")
    )
    args.draft_fit = write(
        tmp_path / "draft_fit.json",
        fit(515840, "expert_conditioned", "draft_unavailable"),
    )
    args.semantic_report = write(
        tmp_path / "semantic.json",
        {
            "status": "complete",
            "counts": {
                "remaining_possible_semantic_repairs": 45,
                "strict_breaks": 10,
            },
        },
    )
    args.output = tmp_path / "comparison.json"
    result = compare(args)
    assert result["gate_pass"] is True
    assert result["holdout_authorized"] is True


def test_expert_identity_gate_fails_when_permutation_does_not_hurt(tmp_path: Path) -> None:
    args = Args()
    arms = {
        "treatment_report": evaluation(300),
        "shared_report": evaluation(250),
        "draft_report": evaluation(247),
        "zero_report": evaluation(280, "zero"),
        "mean_report": evaluation(279, "mean"),
        "permutation_report": evaluation(295, "permutation"),
    }
    for name, report in arms.items():
        setattr(args, name, write(tmp_path / f"{name}.json", report))
    args.unchanged_report = write(tmp_path / "unchanged.json", evaluation(191))
    args.mtr_report = write(tmp_path / "mtr.json", evaluation(204))
    args.treatment_fit = write(
        tmp_path / "treatment_fit.json", fit(515840, "expert_conditioned", "normal")
    )
    args.shared_fit = write(tmp_path / "shared_fit.json", fit(524288, "shared", "normal"))
    args.draft_fit = write(
        tmp_path / "draft_fit.json",
        fit(515840, "expert_conditioned", "draft_unavailable"),
    )
    args.semantic_report = write(
        tmp_path / "semantic.json",
        {
            "status": "complete",
            "counts": {
                "remaining_possible_semantic_repairs": 45,
                "strict_breaks": 10,
            },
        },
    )
    args.output = tmp_path / "comparison.json"
    result = compare(args)
    assert result["gates"]["code_permutation_margin_13"] is False
    assert result["gate_pass"] is False
