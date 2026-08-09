import hashlib
import json
from argparse import Namespace
from pathlib import Path

from compare_mpr3_development import compare


def write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def evaluation(correct: int, math: int, logic: int, code: int) -> dict:
    return {"schema": "shohin-idr1-revision-evaluation-v1", "status": "complete", "split": "development", "full_row_count": 1289, "metrics": {"overall": {"generated_correct": correct}, "math500": {"generated_correct": math}, "bbh_logic": {"generated_correct": logic}, "mbpp": {"generated_correct": code}}}


def fit(data_sha: str, control: str) -> dict:
    return {"schema": "shohin-rme1-product-training-v1", "status": "complete", "updates": 256, "trainable_parameters": 1_179_648, "protected_router_expert_trainables": 0, "data_sha256": data_sha, "rme1_draft_control": control, "rme1_config": {"mode": "shared", "controlled_layers": 16, "rank": 18}}


def make_args(tmp_path: Path, aligned: int = 345) -> Namespace:
    aligned_data = write(tmp_path / "a.jsonl", {"a": 1})
    shuffled_data = write(tmp_path / "s.jsonl", {"s": 1})
    a_sha = hashlib.sha256(aligned_data.read_bytes()).hexdigest()
    s_sha = hashlib.sha256(shuffled_data.read_bytes()).hexdigest()
    return Namespace(
        aligned_report=write(tmp_path / "ar.json", evaluation(aligned, 90, 240, 15)),
        shuffled_report=write(tmp_path / "sr.json", evaluation(330, 85, 233, 12)),
        hidden_report=write(tmp_path / "hr.json", evaluation(329, 84, 233, 12)),
        owner_report=write(tmp_path / "or.json", evaluation(305, 80, 215, 10)),
        aligned_fit=write(tmp_path / "af.json", fit(a_sha, "normal")),
        shuffled_fit=write(tmp_path / "sf.json", fit(s_sha, "normal")),
        hidden_fit=write(tmp_path / "hf.json", fit(a_sha, "draft_unavailable")),
        data_report=write(tmp_path / "data.json", {"schema": "shohin-mpr2-revision-data-report-v1", "status": "complete", "holdout_used": False, "complete_retention": True, "owner_development_score": 305, "outputs": {"train_aligned": {"sha256": a_sha}, "train_shuffled": {"sha256": s_sha}}}),
        semantic_attribution=write(tmp_path / "sem.json", {"schema": "shohin-moe-semantic-repair-attribution-v1", "status": "complete", "counts": {"remaining_possible_semantic_repairs": 30, "strict_breaks": 10}}),
        output=tmp_path / "out.json",
    )


def test_mpr3_gate_passes_only_full_conjunction(tmp_path):
    result = compare(make_args(tmp_path))
    assert result["gate_pass"] is True
    assert result["holdout_authorized"] is True


def test_mpr3_gate_closes_below_owner_margin(tmp_path):
    result = compare(make_args(tmp_path, aligned=343))
    assert result["gate_pass"] is False
    assert result["holdout_authorized"] is False
